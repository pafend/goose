use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, info, warn};

/// Configuration for rate limiting and retry behavior
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Initial backoff delay in milliseconds
    pub initial_backoff_ms: u64,
    /// Maximum backoff delay in milliseconds 
    pub max_backoff_ms: u64,
    /// Backoff multiplier for exponential backoff
    pub backoff_multiplier: f64,
    /// Whether to add jitter to prevent thundering herd
    pub use_jitter: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 1000, // 1 second
            max_backoff_ms: 30000,     // 30 seconds
            backoff_multiplier: 2.0,
            use_jitter: true,
        }
    }
}

impl RateLimitConfig {
    /// Create config optimized for free tier models with more conservative retries
    pub fn free_tier() -> Self {
        Self {
            max_retries: 5,
            initial_backoff_ms: 5000, // 5 seconds for free tier
            max_backoff_ms: 60000,    // 1 minute max
            backoff_multiplier: 2.0,
            use_jitter: true,
        }
    }

    /// Create config for paid tier models with faster retries
    pub fn paid_tier() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 1000, // 1 second for paid tier
            max_backoff_ms: 15000,    // 15 seconds max
            backoff_multiplier: 2.0,
            use_jitter: true,
        }
    }
}

/// Rate limiter with exponential backoff retry logic
#[derive(Debug)]
pub struct RateLimiter {
    config: RateLimitConfig,
}

impl RateLimiter {
    /// Create a new rate limiter with the given configuration
    pub fn new(config: RateLimitConfig) -> Self {
        Self { config }
    }

    /// Create a rate limiter optimized for free tier models
    pub fn for_free_tier() -> Self {
        Self::new(RateLimitConfig::free_tier())
    }

    /// Create a rate limiter optimized for paid tier models
    pub fn for_paid_tier() -> Self {
        Self::new(RateLimitConfig::paid_tier())
    }

    /// Execute a closure with exponential backoff retry on rate limit errors
    /// 
    /// The closure should return Ok(T) on success, or an error that can be checked
    /// for rate limiting using the `is_rate_limit_error` function.
    pub async fn execute_with_retry<T, E, F, Fut, IsRateLimitFn>(
        &self,
        mut operation: F,
        is_rate_limit_error: IsRateLimitFn,
    ) -> Result<T, E>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
        IsRateLimitFn: Fn(&E) -> bool,
        E: std::fmt::Display,
    {
        let mut attempt = 0;
        
        loop {
            match operation().await {
                Ok(result) => {
                    if attempt > 0 {
                        info!("Request succeeded after {} retries", attempt);
                    }
                    return Ok(result);
                }
                Err(error) => {
                    if !is_rate_limit_error(&error) {
                        // Not a rate limit error, return immediately
                        return Err(error);
                    }

                    attempt += 1;
                    if attempt > self.config.max_retries {
                        warn!(
                            "Maximum retry attempts ({}) exceeded for rate limit error: {}",
                            self.config.max_retries, error
                        );
                        return Err(error);
                    }

                    let backoff_ms = self.calculate_backoff_delay(attempt);
                    warn!(
                        "Rate limit error (attempt {}/{}): {}. Retrying in {}ms",
                        attempt, self.config.max_retries, error, backoff_ms
                    );

                    sleep(Duration::from_millis(backoff_ms)).await;
                }
            }
        }
    }

    /// Calculate the backoff delay for a given attempt number
    fn calculate_backoff_delay(&self, attempt: u32) -> u64 {
        let exponential_delay = self.config.initial_backoff_ms
            * (self.config.backoff_multiplier.powi((attempt - 1) as i32) as u64);
        
        let delay = exponential_delay.min(self.config.max_backoff_ms);
        
        if self.config.use_jitter {
            // Add random jitter of ±25% to prevent thundering herd
            let jitter_range = delay / 4; // 25% of delay
            let jitter = fastrand::u64(0..=jitter_range * 2); // 0 to 50% of delay
            let jittered_delay = delay.saturating_sub(jitter_range).saturating_add(jitter);
            debug!("Applied jitter: base delay {}ms, jittered delay {}ms", delay, jittered_delay);
            jittered_delay
        } else {
            delay
        }
    }
}

/// Simple request throttler to prevent hitting rate limits
#[derive(Debug)]
pub struct RequestThrottler {
    /// Minimum time between requests in milliseconds
    min_interval_ms: u64,
    /// Last request timestamp
    last_request: std::sync::Mutex<Option<std::time::Instant>>,
}

impl RequestThrottler {
    /// Create a new request throttler
    pub fn new(min_interval_ms: u64) -> Self {
        Self {
            min_interval_ms,
            last_request: std::sync::Mutex::new(None),
        }
    }

    /// Create a throttler for free tier models (more conservative)
    pub fn for_free_tier() -> Self {
        Self::new(2000) // 2 seconds between requests for free tier
    }

    /// Create a throttler for paid tier models (less restrictive)
    pub fn for_paid_tier() -> Self {
        Self::new(100) // 100ms between requests for paid tier
    }

    /// Wait if necessary to respect the minimum interval between requests
    pub async fn throttle(&self) {
        let now = std::time::Instant::now();
        let should_wait = {
            let mut last_request = self.last_request.lock().unwrap();
            if let Some(last) = *last_request {
                let elapsed = now.duration_since(last);
                let min_interval = Duration::from_millis(self.min_interval_ms);
                if elapsed < min_interval {
                    let wait_time = min_interval - elapsed;
                    debug!("Throttling request for {:?}", wait_time);
                    Some(wait_time)
                } else {
                    *last_request = Some(now);
                    None
                }
            } else {
                *last_request = Some(now);
                None
            }
        };

        if let Some(wait_time) = should_wait {
            sleep(wait_time).await;
            // Update the timestamp after waiting
            let mut last_request = self.last_request.lock().unwrap();
            *last_request = Some(std::time::Instant::now());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_rate_limit_config_defaults() {
        let config = RateLimitConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.initial_backoff_ms, 1000);
        assert_eq!(config.max_backoff_ms, 30000);
        assert_eq!(config.backoff_multiplier, 2.0);
        assert!(config.use_jitter);
    }

    #[test]
    fn test_rate_limit_config_free_tier() {
        let config = RateLimitConfig::free_tier();
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.initial_backoff_ms, 5000);
        assert!(config.max_backoff_ms >= 30000);
    }

    #[test]
    fn test_calculate_backoff_delay() {
        let config = RateLimitConfig {
            max_retries: 3,
            initial_backoff_ms: 1000,
            max_backoff_ms: 10000,
            backoff_multiplier: 2.0,
            use_jitter: false, // Disable jitter for predictable testing
        };
        let limiter = RateLimiter::new(config);

        assert_eq!(limiter.calculate_backoff_delay(1), 1000);  // 1000 * 2^0
        assert_eq!(limiter.calculate_backoff_delay(2), 2000);  // 1000 * 2^1
        assert_eq!(limiter.calculate_backoff_delay(3), 4000);  // 1000 * 2^2
        assert_eq!(limiter.calculate_backoff_delay(4), 8000);  // 1000 * 2^3
        assert_eq!(limiter.calculate_backoff_delay(5), 10000); // Capped at max_backoff_ms
    }

    #[tokio::test]
    async fn test_execute_with_retry_success_first_attempt() {
        let limiter = RateLimiter::for_paid_tier();
        let call_count = Arc::new(AtomicU32::new(0));
        let call_count_clone = Arc::clone(&call_count);

        let result = limiter
            .execute_with_retry(
                || {
                    let count = call_count_clone.clone();
                    async move {
                        count.fetch_add(1, Ordering::SeqCst);
                        Ok::<_, String>("success")
                    }
                },
                |_error: &String| false, // Never consider it a rate limit error
            )
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_execute_with_retry_rate_limit_then_success() {
        let config = RateLimitConfig {
            max_retries: 2,
            initial_backoff_ms: 10, // Very short for testing
            max_backoff_ms: 50,
            backoff_multiplier: 2.0,
            use_jitter: false,
        };
        let limiter = RateLimiter::new(config);
        let call_count = Arc::new(AtomicU32::new(0));
        let call_count_clone = Arc::clone(&call_count);

        let result = limiter
            .execute_with_retry(
                || {
                    let count = call_count_clone.clone();
                    async move {
                        let attempt = count.fetch_add(1, Ordering::SeqCst) + 1;
                        if attempt <= 2 {
                            Err("Rate limit exceeded".to_string())
                        } else {
                            Ok("success")
                        }
                    }
                },
                |error: &String| error.contains("Rate limit"),
            )
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_execute_with_retry_max_attempts_exceeded() {
        let config = RateLimitConfig {
            max_retries: 2,
            initial_backoff_ms: 1, // Very short for testing
            max_backoff_ms: 5,
            backoff_multiplier: 2.0,
            use_jitter: false,
        };
        let limiter = RateLimiter::new(config);
        let call_count = Arc::new(AtomicU32::new(0));
        let call_count_clone = Arc::clone(&call_count);

        let result = limiter
            .execute_with_retry(
                || {
                    let count = call_count_clone.clone();
                    async move {
                        count.fetch_add(1, Ordering::SeqCst);
                        Err::<String, _>("Rate limit exceeded".to_string())
                    }
                },
                |error: &String| error.contains("Rate limit"),
            )
            .await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Rate limit exceeded");
        assert_eq!(call_count.load(Ordering::SeqCst), 3); // Initial + 2 retries
    }

    #[tokio::test]
    async fn test_execute_with_retry_non_rate_limit_error() {
        let limiter = RateLimiter::for_paid_tier();
        let call_count = Arc::new(AtomicU32::new(0));
        let call_count_clone = Arc::clone(&call_count);

        let result = limiter
            .execute_with_retry(
                || {
                    let count = call_count_clone.clone();
                    async move {
                        count.fetch_add(1, Ordering::SeqCst);
                        Err::<String, _>("Other error".to_string())
                    }
                },
                |error: &String| error.contains("Rate limit"),
            )
            .await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Other error");
        assert_eq!(call_count.load(Ordering::SeqCst), 1); // No retries for non-rate-limit errors
    }

    #[tokio::test]
    async fn test_request_throttler() {
        let throttler = RequestThrottler::new(100); // 100ms between requests
        
        let start = std::time::Instant::now();
        throttler.throttle().await; // First request should not wait
        let first_duration = start.elapsed();
        
        throttler.throttle().await; // Second request should wait
        let second_duration = start.elapsed();
        
        // First request should be immediate (< 50ms)
        assert!(first_duration < Duration::from_millis(50));
        // Second request should wait at least 100ms total
        assert!(second_duration >= Duration::from_millis(100));
    }
}
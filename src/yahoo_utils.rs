use std::time::{Duration, UNIX_EPOCH};
use time::OffsetDateTime;
// use tokio_test;
use reqwest::StatusCode;
use yahoo_finance_api as yahoo;

const YCHART_URL: &str = "https://query1.finance.yahoo.com/v8/finance/chart";
const YSEARCH_URL: &str = "https://query2.finance.yahoo.com/v1/finance/search";

macro_rules! YCHART_PERIOD_QUERY {
    () => {
        "{url}/{symbol}?symbol={symbol}&period1={start}&period2={end}&interval={interval}&events=div|split|capitalGains"
    };
}

pub async fn get_quote_history_interval(
    client: &reqwest::Client,
    ticker: &str,
    start: OffsetDateTime,
    end: OffsetDateTime,
    interval: &str,
) -> Result<serde_json::Value, yahoo::YahooError> {
    let url = format!(
        YCHART_PERIOD_QUERY!(),
        url = YCHART_URL,
        symbol = ticker,
        start = start.unix_timestamp(),
        end = end.unix_timestamp(),
        interval = interval
    );
    let resp = client.get(url).send().await?;

    match resp.status() {
        StatusCode::OK => Ok(resp.json().await?),
        status => Err(yahoo::YahooError::FetchFailed(format!("{}", status))),
    }
}

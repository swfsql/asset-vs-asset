#![feature(fs_try_exists)]

// use plotly::common::Mode;
// use plotly::layout::{GridDomain, GridPattern, GridXSide, LayoutGrid, ModeBar, RowOrder};
// use plotly::{Plot, Scatter};
use plotters::prelude::*;
use reqwest::Client;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::path::PathBuf;
use yahoo::Quote;
pub mod yahoo_utils;

// fn line_and_scatter_plot() -> anyhow::Result<()> {
//     let trace1 = Scatter::new(vec![1, 2, 3, 4], vec![10, 15, 13, 17])
//         .name("trace1")
//         .mode(Mode::Markers);
//     let trace2 = Scatter::new(vec![2, 3, 4, 5], vec![16, 5, 11, 9])
//         .name("trace2")
//         .mode(Mode::Lines);
//     let trace3 = Scatter::new(vec![1, 2, 3, 4], vec![12, 9, 15, 12]).name("trace3");

//     let mut plot = Plot::new();
//     plot.add_trace(trace1);
//     plot.add_trace(trace2);
//     plot.add_trace(trace3);
//     // plot.show();

//     let html = plot.to_html();
//     let dir = std::env::var("CARGO_MANIFEST_DIR")?;
//     let path = PathBuf::from(dir).join("out.html");
//     std::fs::write(path, html)?;

//     Ok(())
// }

// fn main() -> anyhow::Result<()> {
//     line_and_scatter_plot()?;
//     Ok(())
// }

use std::time::{Duration, UNIX_EPOCH};
use time::OffsetDateTime;
// use tokio_test;
use yahoo_finance_api as yahoo;

pub trait QuoteExt {
    fn average(&self) -> f64;
    fn adjusted_close(&self) -> f64;
}

impl QuoteExt for Quote {
    fn average(&self) -> f64 {
        (self.open + self.close) / 2.0
    }
    fn adjusted_close(&self) -> f64 {
        self.adjclose
    }
}

pub struct Asset {
    pub metadata: yahoo::YMetaData,
    pub quotes: Vec<Quote>,
    pub dividends: Vec<f64>,
}

impl Asset {
    pub async fn get(client: &Client, ticker: &str) -> anyhow::Result<yahoo::YResponse> {
        let dir = std::env::var("CARGO_MANIFEST_DIR")?;
        let path = PathBuf::from(dir).join(format!("{ticker}.json"));

        let start_utc = time::OffsetDateTime::UNIX_EPOCH;
        let now_utc = time::OffsetDateTime::now_utc();

        let yresp = if std::fs::try_exists(&path)? {
            let yresp = std::fs::read(path)?;
            serde_json::from_slice(&yresp)?
        } else {
            println!("downloading {ticker} data");
            let yresp =
                yahoo_utils::get_quote_history_interval(client, ticker, start_utc, now_utc, "1d")
                    .await?;
            let () = std::fs::write(path, serde_json::to_string_pretty(&yresp)?)?;
            yresp
        };
        Ok(yahoo::YResponse::from_json(yresp)?)
    }

    pub fn fill(yresp: &yahoo::YResponse) -> anyhow::Result<Asset> {
        let quotes = yresp.quotes()?;
        let first_quote = quotes.first().unwrap();
        let mut quotes_fill = vec![first_quote.clone()];
        let mut q = 0;
        let mut current_ts = ts_to_date(first_quote.timestamp);
        loop {
            if q >= quotes.len() {
                break;
            }

            let quote = &quotes[q];
            let quote_ts = ts_to_date(quote.timestamp);

            let diff = if current_ts < quote_ts {
                quote_ts - current_ts
            } else {
                current_ts - quote_ts
            };

            match (
                diff <= Duration::from_secs(24 * 60 * 60),
                current_ts.day() == quote_ts.day(),
                current_ts < quote_ts,
            ) {
                // acquire the next quote from q
                (_close @ true, _same_day @ true, _early @ true) => {
                    current_ts = quote_ts;
                    quotes_fill.push(quote.clone());
                    q += 1;
                    continue;
                }
                // a little after but still on the same day, replaces last quote from q
                (_close @ true, _same_day @ true, _early @ false) => {
                    current_ts = quote_ts;
                    let last_quote = quotes_fill.iter_mut().last().unwrap();
                    *last_quote = quote.clone();
                    q += 1;
                    continue;
                }
                // close but not on the same day, fill info (which may be replaced)
                (_close @ true, _same_day @ false, _early @ true) => {
                    // clone last added and advances a day
                    let mut last_quote_copy = quotes_fill.last().unwrap().clone();
                    // this could be problematic:
                    last_quote_copy.timestamp += 24 * 60 * 60;
                    current_ts = ts_to_date(last_quote_copy.timestamp);
                    quotes_fill.push(last_quote_copy);
                    continue;
                }
                // cannot be close but still be a day later
                (_close @ true, _same_day @ false, _early @ false) => unreachable!(),
                // not close, and whether it's the same day is irrelevant
                // just fills info (too early)
                (_close @ false, _same_day, _early @ true) => {
                    // clone last added and advances a day
                    let mut last_quote_copy = quotes_fill.last().unwrap().clone();
                    // this could be problematic:
                    last_quote_copy.timestamp += 24 * 60 * 60;
                    current_ts = ts_to_date(last_quote_copy.timestamp);
                    quotes_fill.push(last_quote_copy);
                    continue;
                }
                // cannot be not close and after
                (_close @ false, _same_day, _early @ false) => unreachable!(),
            };
        }

        let mut acc_dividend = 1.0;
        let len = quotes_fill.len();
        let d = yresp.dividends()?;
        let mut dividends_fill = vec![];
        let next_dividend = 0;
        for i in 0..len {
            let quote = &quotes_fill[len - 1 - i];
            let quote_ts = ts_to_date(quote.timestamp);
            let next_d = match d.get(next_dividend) {
                Some(next_d) => next_d,
                None => {
                    dividends_fill.push(acc_dividend);
                    continue;
                }
            };
            let dividend_ts = ts_to_date(next_d.date);

            let diff = if dividend_ts < quote_ts {
                quote_ts - dividend_ts
            } else {
                dividend_ts - quote_ts
            };

            match (
                diff <= Duration::from_secs(24 * 60 * 60),
                dividend_ts.day() == quote_ts.day(),
            ) {
                // increase acc_dividend
                (_close @ true, _same_day @ true) => {
                    // TODO: this is wrong because the price already contains
                    // the dividend
                    // also, the dividend value is an addition that happened into the price
                    //
                    // dividend divided by that day's price results in how many
                    // "shares" the owner gets per unit previously invested,
                    // since the dividends are re-invested
                    let add = next_d.amount / (quote.adjusted_close());
                    acc_dividend += add;
                    dividends_fill.push(acc_dividend);
                    continue;
                }
                // fills with the last acc_dividend
                _ => {
                    dividends_fill.push(acc_dividend);
                }
            };
        }
        dividends_fill.reverse();
        assert_eq!(quotes_fill.len(), dividends_fill.len());

        Ok(Asset {
            metadata: yresp.metadata()?,
            quotes: quotes_fill,
            dividends: dividends_fill,
        })
    }

    pub async fn get_and_fill(client: &Client, ticker: &str) -> anyhow::Result<Self> {
        let get = Self::get(client, ticker).await?;
        Self::fill(&get)
    }

    pub fn offset(&self, offset_ts: OffsetDateTime) -> usize {
        self.quotes
            .iter()
            .enumerate()
            .find(|(_i, q)| {
                let ts = ts_to_date(q.timestamp);
                offset_ts.year() == ts.year()
                    && offset_ts.month() == ts.month()
                    && offset_ts.day() == ts.day()
            })
            .map(|(i, _q)| i)
            .unwrap()
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("start");
    let client = Client::default();

    const BTC: &str = "BTC-USD";
    const PETR4: &str = "PETR4.SA";

    let mut assets = HashMap::new();
    assets.insert(BTC.to_string(), Asset::get_and_fill(&client, BTC).await?);
    assets.insert(
        PETR4.to_string(),
        Asset::get_and_fill(&client, PETR4).await?,
    );

    let currencies: Vec<String> = assets
        .values()
        .map(|asset| asset.metadata.currency.clone())
        .collect();
    for currency in currencies {
        if &currency == "USD" {
            continue;
        }
        match assets.entry(currency.clone()) {
            Entry::Occupied(_) => (),
            Entry::Vacant(entry) => {
                let info = format!("{currency}=X");
                entry.insert(Asset::get_and_fill(&client, &info).await?);
            }
        }
    }
    println!("data loaded");

    let mut offset_ts = 0;
    let mut take = usize::MAX;
    for asset in assets.values() {
        let first = asset.quotes.first().unwrap();
        offset_ts = offset_ts.max(first.timestamp);
        take = take.min(asset.quotes.len());
    }
    let offset_ts = ts_to_date(offset_ts);

    let btc = &assets.get(BTC).unwrap();
    let petr4 = &assets.get(PETR4).unwrap();
    let brl_by_usd = &assets.get("BRL").unwrap();

    let btc_offset = btc.offset(offset_ts);
    let petr4_offset = petr4.offset(offset_ts);
    let brl_by_usd_offset = brl_by_usd.offset(offset_ts);

    let btc = &btc.quotes[btc_offset..btc_offset + take];
    let petr4 = &petr4.quotes[petr4_offset..petr4_offset + take];
    let brl_by_usd = &brl_by_usd.quotes[brl_by_usd_offset..brl_by_usd_offset + take];

    assert_eq!(btc.len(), petr4.len());
    assert_eq!(btc.len(), brl_by_usd.len());

    let last_btc = btc.last().unwrap().adjusted_close();
    let last_petr4 = petr4.last().unwrap().adjusted_close();
    let last_brl_by_usd = brl_by_usd.last().unwrap().adjusted_close();

    let btc_gain: Vec<(f64, u64)> = btc
        .iter()
        .map(|q| (last_btc / q.adjusted_close(), q.timestamp))
        .collect();
    let petr4_gain: Vec<(f64, u64)> = petr4
        .iter()
        .map(|q| (last_petr4 / q.adjusted_close(), q.timestamp))
        .collect();
    let brl_by_usd_gain: Vec<(f64, u64)> = brl_by_usd
        .iter()
        .map(|q| (last_brl_by_usd / q.adjusted_close(), q.timestamp))
        .collect();
    let petr4_gain_usd: Vec<(f64, u64)> = petr4_gain
        .iter()
        .zip(brl_by_usd_gain)
        .map(|(petr4, brl_by_usd)| (petr4.0 / brl_by_usd.0, petr4.1))
        .collect();

    /*
        let Oklab { l, a, b } = oklab::srgb_to_oklab(oklab::RGB {
            r: p.0[0],
            g: p.0[1],
            b: p.0[2],
        });

        // 3. 2D Lab -> 2D Lch (lightness, chroma, hue)
        let c = (a * a + b * b).sqrt();
        let h = f32::atan2(b, a);

        let lab: Vec<_> = lc_2d
        .into_iter()
        .zip(h_2d)
        .map(|((l, c), h)| {
            let a = c.0 * h.0.cos();
            let b = c.0 * h.0.sin();
            oklab::Oklab { l: l.0, a, b }
        })
        .collect();

    // 15. 2D Lab -> 2D rgb
    let rgb: Vec<_> = lab.into_iter().map(oklab::oklab_to_srgb).collect();

        */

    let mut halvings = vec![
        time::Date::from_calendar_date(2012, time::Month::November, 28)?,
        time::Date::from_calendar_date(2016, time::Month::July, 9)?,
        time::Date::from_calendar_date(2020, time::Month::May, 11)?,
    ];
    // expected next halving
    halvings.push(time::Date::from_calendar_date(
        2024,
        time::Month::April,
        15,
    )?);
    // expected future halvings
    halvings.push(halvings.last().cloned().unwrap() + Duration::from_secs(210000 * 10 * 60));
    halvings.push(halvings.last().cloned().unwrap() + Duration::from_secs(210000 * 10 * 60));
    halvings.push(halvings.last().cloned().unwrap() + Duration::from_secs(210000 * 10 * 60));
    halvings.push(halvings.last().cloned().unwrap() + Duration::from_secs(210000 * 10 * 60));

    let btc_gain_len = btc_gain.len();
    let x: Vec<(f64, RGBColor)> = btc_gain
        .iter()
        .zip(petr4_gain_usd)
        .enumerate()
        .map(|(i, (btc, petr4))| {
            let color = {
                let lightness = 0.560;
                let chroma = 0.26;
                let date = ts_to_date(btc.1).date();
                let (next_halving_i, next_halving) = halvings
                    .iter()
                    .enumerate()
                    .find(|(j, h)| **h > date)
                    .unwrap();
                let previous_halving = halvings[next_halving_i - 1];
                let halving_progress =
                    (date - previous_halving) / (*next_halving - previous_halving);
                let hue = halving_progress * 360.0;
                let rgb = oklab::oklab_to_srgb(oklab::Oklab {
                    l: lightness,
                    a: chroma * hue.cos() as f32,
                    b: chroma * hue.sin() as f32,
                });
                RGBColor(rgb.r, rgb.g, rgb.b)
            };
            if btc.0 > petr4.0 {
                (btc.0 / petr4.0, RGBColor(0x00, 0x00, 0xff))
            } else if petr4.0 > btc.0 {
                (-petr4.0 / btc.0, RGBColor(0xff, 0x00, 0x00))
            } else {
                (1.0, RGBColor(0x00, 0x00, 0x00))
            }
        })
        .collect();

    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for x in &x {
        if x.0 < min {
            min = x.0;
        }
        if x.0 > max {
            max = x.0;
        }
    }

    let one = 1f64.to_bits();
    let x_btc: Vec<(f64, RGBColor)> = x
        .iter()
        .filter(|x| x.0.is_sign_positive())
        .cloned()
        .collect();
    let x_petr4: Vec<(f64, RGBColor)> = x
        .iter()
        .filter_map(|x| {
            if x.0.is_sign_negative() {
                Some((-x.0, x.1))
            } else if x.0.to_bits() == one {
                Some((x.0, x.1))
            } else {
                None
            }
        })
        .collect();
    println!("min: {min}; max: {max}; len: {take}");

    let width = 1920;
    let height = 1080;

    let bin_len = 800;
    // let max_abs = max.abs().max(min.abs());
    // let range = max_abs;
    let range = 80.0;
    let linear_step = range / bin_len as f64;

    let mut right_asset = vec![];
    let mut right_asset_heights = vec![0usize; bin_len + 1];
    let mut left_asset = vec![];
    let mut left_asset_heights = vec![0usize; bin_len + 1];
    for x in x_btc.iter() {
        let i = (x.0 / linear_step) as usize;
        right_asset_heights[i] += 1;
        right_asset.push(Rectangle::new(
            [
                (
                    right_asset_heights[i] as i32, /* * bin_width as i32*/
                    (i as f64 * linear_step),      /* * bin_height as u32*/
                ),
                (
                    right_asset_heights[i] as i32 - 1, /* * bin_width as i32*/
                    ((i + 1) as f64 * linear_step),    /* * bin_height as u32*/
                ),
            ],
            ShapeStyle {
                color: x.1.into(),
                filled: true,
                stroke_width: 1,
            },
        ));
    }
    for x in x_petr4.iter() {
        let i = (x.0 / linear_step) as usize;
        left_asset_heights[i] += 1;
        left_asset.push(Rectangle::new(
            [
                (
                    -(left_asset_heights[i] as i32), /* * bin_width as i32*/
                    (i as f64 * linear_step),        /* * bin_height as u32*/
                ),
                (
                    -(left_asset_heights[i] as i32 - 1), /* * bin_width as i32*/
                    ((i + 1) as f64 * linear_step),      /* * bin_height as u32*/
                ),
            ],
            ShapeStyle {
                color: x.1.into(),
                filled: true,
                stroke_width: 1,
            },
        ));
    }

    {
        let root =
            BitMapBackend::new("../plot.png", (width as u32, height as u32)).into_drawing_area();
        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&root)
            .caption("PETR4 vs BTC    ", ("sans-serif", 50).into_font())
            .margin(5)
            .top_x_label_area_size(30)
            .right_y_label_area_size(30)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d((-140i32..140i32), (80f64..1f64).log_scale())?;

        chart
            .configure_mesh()
            .x_labels(40)
            .y_labels(80)
            .x_label_formatter(&|v| format!("{} dias", v.abs()))
            .y_label_formatter(&|v| format!("{v:}x"))
            // .x_desc("Dias")
            // .y_desc("Ganho")
            // .disable_y_mesh()
            .y_max_light_lines(10)
            .draw()?;

        chart.draw_series(right_asset)?;
        // .label("BTC gain / PETR4 gain");
        chart.draw_series(left_asset)?;
        // .label("PETR4 gain / BTC gain");

        // chart
        //     .draw_series(LineSeries::new(
        //         (-50..=50).map(|x| (x as f32, x as f32)),
        //         &RED,
        //     ))?
        //     .label("y = x^2")
        //     .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        // let hist = Histogram::vertical(&chart);
        // let hist = hist
        //     .style(RED.filled())
        //     .margin(0)
        //     .data(x_btc.iter().map(|x| (x, 0.01)));
        // chart.draw_series(hist)?;

        // chart
        //     .configure_series_labels()
        //     .background_style(&WHITE.mix(0.8))
        //     .border_style(&BLACK)
        //     .draw()?;

        root.present()?;
    }

    // use plotly::box_plot::{BoxMean, BoxPoints};
    // use plotly::common::{ErrorData, ErrorType, Line, Marker, Mode, Orientation, Title};
    // use plotly::histogram::{Bins, Cumulative, HistFunc, HistNorm};
    // use plotly::layout::{Axis, BarMode, BoxMode, Layout, Margin};
    // use plotly::{
    //     color::{NamedColor, Rgb, Rgba},
    //     Bar, BoxPlot, Histogram, Plot, Scatter,
    // };

    // let trace_btc = Histogram::new_vertical(x_btc)
    //     // .marker(Marker::new().color(NamedColor::Pink))
    //     .name("btc");
    // let trace_petr4 = Histogram::new_vertical(x_petr4)
    //     // .marker(Marker::new().color(NamedColor::Pink))
    //     .name("petr4");

    // // let trace = Histogram::new_vertical(x)
    // //     // .marker(Marker::new().color(NamedColor::Pink))
    // //     .name("h");
    // let mut plot = Plot::new();
    // plot.add_trace(trace_btc);
    // plot.add_trace(trace_petr4);

    // let layout = Layout::new()
    //     .grid(
    //         LayoutGrid::new()
    //             .rows(1)
    //             .columns(1)
    //             .pattern(GridPattern::Independent),
    //     )
    //     .mode_bar(ModeBar::new().orientation(Orientation::Horizontal))
    //     .bar_mode(BarMode::Overlay);
    // plot.set_layout(layout);

    // let html = plot.to_html();
    // let dir = std::env::var("CARGO_MANIFEST_DIR")?;
    // let path = PathBuf::from(dir).join("..").join("out.html");
    // std::fs::write(path, html)?;

    println!("end");
    Ok(())
}

fn ts_to_date(ts: u64) -> OffsetDateTime {
    OffsetDateTime::from(UNIX_EPOCH + Duration::from_secs(ts))
}

/*
YMetaData {
    currency: "USD",
    symbol: "AAPL",
    exchange_name: "NMS",
    instrument_type: "EQUITY",
    first_trade_date: Some(
        345479400,
    ),
    regular_market_time: 1709758801,
    gmtoffset: -18000,
    timezone: "EST",
    exchange_timezone_name: "America/New_York",
    regular_market_price: 169.12,
    chart_previous_close: 189.3,
    previous_close: None,
    scale: None,
    price_hint: 2,
    current_trading_period: TradingPeriod {
        pre: PeriodInfo {
            timezone: "EST",
            start: 1709802000,
            end: 1709821800,
            gmtoffset: -18000,
        },
        regular: PeriodInfo {
            timezone: "EST",
            start: 1709821800,
            end: 1709845200,
            gmtoffset: -18000,
        },
        post: PeriodInfo {
            timezone: "EST",
            start: 1709845200,
            end: 1709859600,
            gmtoffset: -18000,
        },
    },
    trading_periods: None,
    data_granularity: "1d",
    range: "1mo",
    valid_ranges: [
        "1d",
        "5d",
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "2y",
        "5y",
        "10y",
        "ytd",
        "max",
    ],
}

*/

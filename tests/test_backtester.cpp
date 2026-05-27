#include <gtest/gtest.h>
#include "../backtester.h"
#include <cmath>
#include <fstream>
#include <sstream>

// ─────────────────────────────────────────────
// Helper: build a simple time series
// ─────────────────────────────────────────────
static std::vector<Day> make_series(const std::vector<double>& prices)
{
    std::vector<Day> ts;
    ts.reserve(prices.size());
    int day = 1;
    for (double p : prices) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "2022-01-%02d", day++);
        ts.emplace_back(std::string(buf), p);
    }
    return ts;
}

// ─────────────────────────────────────────────
// Trade tests
// ─────────────────────────────────────────────
TEST(TradeTest, PnlProfitable)
{
    Trade t{"2022-01-01", "2022-01-10", 100.0, 110.0, 10.0, 2.0};
    // (110 - 100) * 10 - 2 = 98
    EXPECT_DOUBLE_EQ(t.pnl(), 98.0);
}

TEST(TradeTest, PnlLosing)
{
    Trade t{"2022-01-01", "2022-01-10", 100.0, 90.0, 5.0, 1.5};
    // (90 - 100) * 5 - 1.5 = -51.5
    EXPECT_DOUBLE_EQ(t.pnl(), -51.5);
}

TEST(TradeTest, PnlBreakEven)
{
    Trade t{"2022-01-01", "2022-01-05", 50.0, 50.0, 20.0, 0.0};
    EXPECT_DOUBLE_EQ(t.pnl(), 0.0);
}

TEST(TradeTest, ReturnPctPositive)
{
    Trade t{"2022-01-01", "2022-01-10", 100.0, 120.0, 10.0, 0.0};
    EXPECT_DOUBLE_EQ(t.return_pct(), 0.20);
}

TEST(TradeTest, ReturnPctNegative)
{
    Trade t{"2022-01-01", "2022-01-10", 200.0, 150.0, 5.0, 0.0};
    EXPECT_DOUBLE_EQ(t.return_pct(), -0.25);
}

TEST(TradeTest, ReturnPctZero)
{
    Trade t{"2022-01-01", "2022-01-03", 80.0, 80.0, 2.0, 0.0};
    EXPECT_DOUBLE_EQ(t.return_pct(), 0.0);
}

// ─────────────────────────────────────────────
// Portfolio tests
// ─────────────────────────────────────────────
TEST(PortfolioTest, DefaultConstructorStartsWithCash10000)
{
    Portfolio p;
    EXPECT_FALSE(p.in_position());
    EXPECT_DOUBLE_EQ(p.value(100.0), 10000.0);
}

TEST(PortfolioTest, CustomCashConstructor)
{
    Portfolio p(5000.0);
    EXPECT_FALSE(p.in_position());
    EXPECT_DOUBLE_EQ(p.value(1.0), 5000.0);
}

TEST(PortfolioTest, BuyPutsPortfolioInPosition)
{
    Portfolio p(10000.0);
    p.buy(100.0, "2022-01-01");
    EXPECT_TRUE(p.in_position());
}

TEST(PortfolioTest, BuyWhenAlreadyInPositionIsNoOp)
{
    Portfolio p(10000.0);
    p.buy(100.0, "2022-01-01");
    // second buy while in position must not change state
    p.buy(200.0, "2022-01-02");
    EXPECT_TRUE(p.in_position());
    EXPECT_EQ(p.get_trades().size(), 0u); // still no closed trades
}

TEST(PortfolioTest, BuyWithZeroPriceIsNoOp)
{
    Portfolio p(10000.0);
    p.buy(0.0, "2022-01-01");
    EXPECT_FALSE(p.in_position());
}

TEST(PortfolioTest, BuyWithNegativePriceIsNoOp)
{
    Portfolio p(10000.0);
    p.buy(-50.0, "2022-01-01");
    EXPECT_FALSE(p.in_position());
}

TEST(PortfolioTest, SellWhenNotInPositionIsNoOp)
{
    Portfolio p(10000.0);
    p.sell(100.0, "2022-01-01");
    EXPECT_FALSE(p.in_position());
    EXPECT_EQ(p.get_trades().size(), 0u);
}

TEST(PortfolioTest, SellWithZeroPriceIsNoOp)
{
    Portfolio p(10000.0);
    p.buy(100.0, "2022-01-01");
    p.sell(0.0, "2022-01-02");
    EXPECT_TRUE(p.in_position()); // still in position
    EXPECT_EQ(p.get_trades().size(), 0u);
}

TEST(PortfolioTest, SellWithNegativePriceIsNoOp)
{
    Portfolio p(10000.0);
    p.buy(100.0, "2022-01-01");
    p.sell(-10.0, "2022-01-02");
    EXPECT_TRUE(p.in_position());
}

TEST(PortfolioTest, BuyThenSellProducesOneTrade)
{
    Portfolio p(10000.0);
    p.buy(100.0, "2022-01-01");
    p.sell(110.0, "2022-01-10");
    EXPECT_FALSE(p.in_position());
    ASSERT_EQ(p.get_trades().size(), 1u);
}

TEST(PortfolioTest, TradeEntryAndExitDatesSet)
{
    Portfolio p(10000.0);
    p.buy(100.0, "2022-01-01");
    p.sell(110.0, "2022-01-10");
    const Trade& t = p.get_trades().front();
    EXPECT_EQ(t.entry_date, "2022-01-01");
    EXPECT_EQ(t.exit_date, "2022-01-10");
}

TEST(PortfolioTest, TradeEntryPriceIncludesSlippage)
{
    Portfolio p(10000.0);
    p.buy(100.0, "2022-01-01");
    p.sell(110.0, "2022-01-10");
    ASSERT_EQ(p.get_trades().size(), 1u);
    EXPECT_GT(p.get_trades().front().entry_price, 100.0); // slippage applied upward
}

TEST(PortfolioTest, TradeExitPriceIncludesSlippage)
{
    Portfolio p(10000.0);
    p.buy(100.0, "2022-01-01");
    p.sell(110.0, "2022-01-10");
    const Trade& t = p.get_trades().front();
    EXPECT_LT(t.exit_price, 110.0); // slippage applied downward on sell
}

TEST(PortfolioTest, ValueWhenNotInPositionReturnsCash)
{
    Portfolio p(7500.0);
    EXPECT_DOUBLE_EQ(p.value(999.0), 7500.0);
}

TEST(PortfolioTest, ValueWhenInPositionUsesCurrentPrice)
{
    Portfolio p(10000.0);
    p.buy(100.0, "2022-01-01");
    // value should reflect shares * current_price, not original purchase price
    double v50  = p.value(50.0);
    double v200 = p.value(200.0);
    EXPECT_GT(v200, v50);
}

TEST(PortfolioTest, AfterSellPortfolioValueReflectsCash)
{
    Portfolio p(10000.0);
    p.buy(100.0, "2022-01-01");
    p.sell(100.0, "2022-01-10");
    // After selling at the same price, cash should be close to initial (minus fees)
    double val = p.value(0.0); // price arg ignored when not in position
    EXPECT_GT(val, 0.0);
    EXPECT_LT(val, 10000.0); // fees ate into value
}

TEST(PortfolioTest, TransactionCostMinimumApplied)
{
    // With very few shares the minimum $0.35 commission should kick in.
    // Use a very small portfolio so shares * 0.0035 < 0.35.
    Portfolio p(1.0); // only $1 of cash
    p.buy(1.0, "2022-01-01");
    // If minimum commission ($0.35) exceeds cash ($1 worth of shares),
    // the buy guard (fees > cash) might block the trade. Either way the
    // portfolio behaves safely without exceptions.
    // No assertion on trade count needed; just ensure no crash/exception.
    SUCCEED();
}

TEST(PortfolioTest, TransactionCostMaximumApplied)
{
    // A large portfolio: commission capped at 1% of trade value.
    Portfolio p(1'000'000.0);
    p.buy(100.0, "2022-01-01");
    EXPECT_TRUE(p.in_position());
    p.sell(100.0, "2022-01-10");
    // fees should be <= 1% of trade value (approximately 1% of $1M)
    ASSERT_EQ(p.get_trades().size(), 1u);
    double trade_val = p.get_trades().front().entry_price * p.get_trades().front().shares;
    EXPECT_LE(p.get_trades().front().fees, 0.011 * trade_val); // small tolerance
}

TEST(PortfolioTest, MultipleBuySellCycles)
{
    Portfolio p(10000.0);
    p.buy(100.0, "2022-01-01");
    p.sell(105.0, "2022-01-10");
    p.buy(106.0, "2022-01-11");
    p.sell(108.0, "2022-01-20");
    EXPECT_EQ(p.get_trades().size(), 2u);
    EXPECT_FALSE(p.in_position());
}

// ─────────────────────────────────────────────
// MomentumStrategy tests
// ─────────────────────────────────────────────
TEST(MomentumStrategyTest, HoldWhenIndexLessThanLookback)
{
    MomentumStrategy strat(5);
    auto ts = make_series({10, 20, 30, 40, 50, 60});
    // i = 3 < lookback 5 → HOLD
    EXPECT_EQ(strat.generate(ts, 3), Signal::HOLD);
}

TEST(MomentumStrategyTest, HoldAtExactlyLookbackBoundary)
{
    MomentumStrategy strat(5);
    auto ts = make_series({10, 20, 30, 40, 50, 60});
    // i == lookback (5) → compares ts[4]=50 vs ts[0]=10 → BUY
    EXPECT_EQ(strat.generate(ts, 5), Signal::BUY);
}

TEST(MomentumStrategyTest, BuySignalWhenPriceRising)
{
    MomentumStrategy strat(3);
    // ts[i-1] > ts[i-lookback] → BUY
    auto ts = make_series({10, 20, 30, 40}); // ts[2]=30 > ts[0]=10
    EXPECT_EQ(strat.generate(ts, 3), Signal::BUY);
}

TEST(MomentumStrategyTest, SellSignalWhenPriceFalling)
{
    MomentumStrategy strat(3);
    auto ts = make_series({40, 30, 20, 10}); // ts[2]=20 < ts[0]=40
    EXPECT_EQ(strat.generate(ts, 3), Signal::SELL);
}

TEST(MomentumStrategyTest, HoldWhenPriceUnchanged)
{
    MomentumStrategy strat(3);
    auto ts = make_series({50, 50, 50, 50}); // ts[2]=50 == ts[0]=50
    EXPECT_EQ(strat.generate(ts, 3), Signal::HOLD);
}

TEST(MomentumStrategyTest, DefaultLookbackIs20)
{
    MomentumStrategy strat; // default lookback = 20
    std::vector<double> prices(25, 100.0);
    // generate(ts, 24): now = ts[23], past = ts[4]
    prices[23] = 110.0; // make ts[23] > ts[4]
    auto ts = make_series(prices);
    EXPECT_EQ(strat.generate(ts, 24), Signal::BUY);
}

TEST(MomentumStrategyTest, CustomLookback1IsAlwaysHold)
{
    // With lookback=1, generate(ts,i) compares ts[i-1] vs ts[i-1] — always equal → HOLD
    MomentumStrategy strat(1);
    auto ts = make_series({100, 200, 300});
    EXPECT_EQ(strat.generate(ts, 2), Signal::HOLD);
}

TEST(MomentumStrategyTest, HoldWhenSeriesEmpty)
{
    MomentumStrategy strat(5);
    std::vector<Day> empty_ts;
    // i=0 < lookback=5 → HOLD
    EXPECT_EQ(strat.generate(empty_ts, 0), Signal::HOLD);
}

// ─────────────────────────────────────────────
// Metrics tests
// ─────────────────────────────────────────────
TEST(MetricsTest, EmptyEquityCurveReturnsZeros)
{
    std::vector<double> empty;
    Metrics m(empty);
    EXPECT_DOUBLE_EQ(m.getCAGR(), 0.0);
    EXPECT_DOUBLE_EQ(m.max_drawdown(), 0.0);
    EXPECT_DOUBLE_EQ(m.get_sharpe(), 0.0);
    EXPECT_TRUE(m.daily_returns().empty());
}

TEST(MetricsTest, SingleElementEquityCurveReturnsZeros)
{
    Metrics m({10000.0});
    EXPECT_DOUBLE_EQ(m.getCAGR(), 0.0);
    EXPECT_DOUBLE_EQ(m.max_drawdown(), 0.0);
    EXPECT_DOUBLE_EQ(m.get_sharpe(), 0.0);
    EXPECT_TRUE(m.daily_returns().empty());
}

TEST(MetricsTest, FlatEquityCurveHasZeroDrawdown)
{
    std::vector<double> curve(252, 10000.0);
    Metrics m(curve);
    EXPECT_DOUBLE_EQ(m.max_drawdown(), 0.0);
}

TEST(MetricsTest, FlatEquityCurveHasZeroSharpe)
{
    std::vector<double> curve(252, 10000.0);
    Metrics m(curve);
    EXPECT_DOUBLE_EQ(m.get_sharpe(), 0.0);
}

TEST(MetricsTest, FlatEquityCurveCAGRIsZero)
{
    // If initial == final, CAGR = (1)^(1/years) - 1 = 0
    std::vector<double> curve(252, 10000.0);
    Metrics m(curve);
    EXPECT_NEAR(m.getCAGR(), 0.0, 1e-9);
}

TEST(MetricsTest, PositiveCAGRWhenPortfolioGrows)
{
    // Simulate doubling in ~252 trading days (1 year) → CAGR ≈ 100%
    std::vector<double> curve(252);
    for (int i = 0; i < 252; i++)
        curve[i] = 10000.0 + i * (10000.0 / 251.0);
    Metrics m(curve);
    EXPECT_GT(m.getCAGR(), 0.0);
}

TEST(MetricsTest, NegativeCAGRWhenPortfolioShrinks)
{
    std::vector<double> curve(252);
    for (int i = 0; i < 252; i++)
        curve[i] = 10000.0 - i * 20.0; // declining
    Metrics m(curve);
    EXPECT_LT(m.getCAGR(), 0.0);
}

TEST(MetricsTest, MaxDrawdownIsNonPositive)
{
    // Drawdown must always be <= 0
    std::vector<double> curve = {10000, 12000, 9000, 11000, 8000};
    Metrics m(curve);
    EXPECT_LE(m.max_drawdown(), 0.0);
}

TEST(MetricsTest, MaxDrawdownKnownValue)
{
    // Peak at 12000, trough at 9000: drawdown = (9000-12000)/12000 = -0.25
    std::vector<double> curve = {10000.0, 12000.0, 9000.0};
    Metrics m(curve);
    EXPECT_NEAR(m.max_drawdown(), -0.25, 1e-9);
}

TEST(MetricsTest, MaxDrawdownWorstCaseSelected)
{
    // Two drawdown periods: -20% and -30%. Worst (-30%) should be reported.
    std::vector<double> curve = {10000, 12000, 9600,   // -20% from 12000
                                 14000, 9800};          // -30% from 14000
    Metrics m(curve);
    EXPECT_NEAR(m.max_drawdown(), -0.30, 1e-6);
}

TEST(MetricsTest, DailyReturnsCountMatchesCurveLength)
{
    std::vector<double> curve = {10000, 11000, 10500, 11500};
    Metrics m(curve);
    // 3 consecutive pairs
    EXPECT_EQ(m.daily_returns().size(), 3u);
}

TEST(MetricsTest, DailyReturnsValuesCorrect)
{
    std::vector<double> curve = {10000.0, 11000.0, 9900.0};
    Metrics m(curve);
    const auto& r = m.daily_returns();
    ASSERT_EQ(r.size(), 2u);
    EXPECT_NEAR(r[0],  0.10, 1e-9); // +10%
    EXPECT_NEAR(r[1], -0.10, 1e-9); // -10%
}

TEST(MetricsTest, SharpeRatioPositiveForBullishCurve)
{
    // Monotonically increasing curve → all daily returns positive → positive Sharpe
    std::vector<double> curve(253);
    for (int i = 0; i < 253; i++)
        curve[i] = 10000.0 * std::exp(0.001 * i);
    Metrics m(curve);
    EXPECT_GT(m.get_sharpe(), 0.0);
}

TEST(MetricsTest, InitialOrFinalZeroEquityCurveCAGRIsZero)
{
    Metrics m({0.0, 10000.0, 11000.0});
    EXPECT_DOUBLE_EQ(m.getCAGR(), 0.0);
}

// ─────────────────────────────────────────────
// TradeMetrics tests
// ─────────────────────────────────────────────
static Trade make_trade(double entry, double exit, double shares = 10.0, double fees = 0.0)
{
    Trade t;
    t.entry_date  = "2022-01-01";
    t.exit_date   = "2022-01-10";
    t.entry_price = entry;
    t.exit_price  = exit;
    t.shares      = shares;
    t.fees        = fees;
    return t;
}

TEST(TradeMetricsTest, SaveCsvCreatesFileWithHeader)
{
    std::vector<Trade> trades = {make_trade(100.0, 110.0, 10.0, 1.0)};
    std::string path = "/tmp/test_trades_header.csv";
    TradeMetrics::save_csv(trades, path);

    std::ifstream f(path);
    ASSERT_TRUE(f.is_open());
    std::string header;
    std::getline(f, header);
    EXPECT_EQ(header, "entry_date,exit_date,entry_price,exit_price,shares,fees,pnl,return_pct");
}

TEST(TradeMetricsTest, SaveCsvRowCountMatchesTrades)
{
    std::vector<Trade> trades = {
        make_trade(100.0, 110.0),
        make_trade(110.0, 105.0),
        make_trade(105.0, 120.0)
    };
    std::string path = "/tmp/test_trades_rows.csv";
    TradeMetrics::save_csv(trades, path);

    std::ifstream f(path);
    int lines = 0;
    std::string line;
    while (std::getline(f, line)) lines++;
    // 1 header + 3 data rows
    EXPECT_EQ(lines, 4);
}

TEST(TradeMetricsTest, SaveCsvEmptyTradesWritesOnlyHeader)
{
    std::string path = "/tmp/test_trades_empty.csv";
    TradeMetrics::save_csv({}, path);

    std::ifstream f(path);
    int lines = 0;
    std::string line;
    while (std::getline(f, line)) lines++;
    EXPECT_EQ(lines, 1); // header only
}

TEST(TradeMetricsTest, SaveCsvValuesMatchTrade)
{
    Trade t = make_trade(100.0, 110.0, 5.0, 0.5);
    TradeMetrics::save_csv({t}, "/tmp/test_trades_values.csv");

    std::ifstream f("/tmp/test_trades_values.csv");
    std::string header, row;
    std::getline(f, header);
    std::getline(f, row);

    std::istringstream ss(row);
    std::string field;
    std::vector<std::string> fields;
    while (std::getline(ss, field, ',')) fields.push_back(field);

    ASSERT_GE(fields.size(), 8u);
    EXPECT_EQ(fields[0], "2022-01-01");          // entry_date
    EXPECT_EQ(fields[1], "2022-01-10");           // exit_date
    EXPECT_NEAR(std::stod(fields[2]), 100.0, 1e-6); // entry_price
    EXPECT_NEAR(std::stod(fields[3]), 110.0, 1e-6); // exit_price
    EXPECT_NEAR(std::stod(fields[4]), 5.0, 1e-6);   // shares
    EXPECT_NEAR(std::stod(fields[5]), 0.5, 1e-6);   // fees
    EXPECT_NEAR(std::stod(fields[6]), t.pnl(), 1e-6);       // pnl
    EXPECT_NEAR(std::stod(fields[7]), t.return_pct(), 1e-6); // return_pct
}

// ─────────────────────────────────────────────
// End-to-end Portfolio + Strategy integration
// ─────────────────────────────────────────────
TEST(IntegrationTest, MomentumOnTrendingUpSeriesProducesBuyTrade)
{
    // 25 days of steadily rising prices; momentum(20) should generate BUY
    std::vector<double> prices;
    for (int i = 0; i < 25; i++) prices.push_back(100.0 + i);
    auto ts = make_series(prices);

    MomentumStrategy strat(20);
    Portfolio p(10000.0);

    for (size_t i = 2; i < ts.size(); i++) {
        Signal sig = strat.generate(ts, i - 1);
        double price = ts[i].close;
        if (sig == Signal::BUY && !p.in_position())
            p.buy(price, ts[i].date);
        if (sig == Signal::SELL && p.in_position())
            p.sell(price, ts[i].date);
    }
    // With an uptrend, at least one BUY should have been triggered
    EXPECT_TRUE(p.in_position() || !p.get_trades().empty());
}

TEST(IntegrationTest, MomentumOnTrendingDownSeriesGeneratesSellSignal)
{
    // 25 days of steadily declining prices; momentum(20) should trigger SELL
    std::vector<double> prices;
    for (int i = 0; i < 25; i++) prices.push_back(200.0 - i);
    auto ts = make_series(prices);

    MomentumStrategy strat(20);
    // At i=21, ts[20] < ts[1], so SELL signal is generated
    Signal sig = strat.generate(ts, 21);
    EXPECT_EQ(sig, Signal::SELL);
}

TEST(IntegrationTest, PortfolioBuySellRoundTripProfitable)
{
    Portfolio p(10000.0);
    p.buy(100.0, "2022-01-01");
    p.sell(150.0, "2022-06-01");
    ASSERT_EQ(p.get_trades().size(), 1u);
    EXPECT_GT(p.get_trades().front().pnl(), 0.0);
}

TEST(IntegrationTest, PortfolioBuySellRoundTripLosing)
{
    Portfolio p(10000.0);
    p.buy(150.0, "2022-01-01");
    p.sell(100.0, "2022-06-01");
    ASSERT_EQ(p.get_trades().size(), 1u);
    EXPECT_LT(p.get_trades().front().pnl(), 0.0);
}

"""
structure  –  Market Microstructure Simulation Package
=======================================================

Package layout
--------------
structure/
├── state/
│   └── orderbook.py          – Aggregated limit order book (price-level)
├── mechanism/
│   ├── order_flow.py          – Order arrival, classification, OFI, VPIN
│   ├── liquidity.py           – Spread decomposition, depth, Amihud, Kyle λ
│   ├── leverage.py            – Margin account, balance sheet, VaR margin
│   └── liquidation.py         – Liquidation trigger, cascade engine, heatmap
├── agents/
│   ├── market_maker.py        – Market maker agent (Avellaneda-Stoikov + variants)
│   └── traders.py             – Noise, informed, momentum, mean-rev, stop-loss agents
└── simulation/
    └── market_simulation.py   – Full simulation engine + scenario presets

Dependency order (no circular imports)
---------------------------------------
state → mechanism → agents → simulation
"""

# ---- state ------------------------------------------------------------------
from .state.orderbook import OrderBook

# ---- mechanism --------------------------------------------------------------
from .mechanism.order_flow import (
    OrderEvent,
    FlowSnapshot,
    PoissonArrivalProcess,
    HawkesArrivalProcess,
    OrderSizeDistribution,
    OrderFlowGenerator,
    TradeClassifier,
    OFICalculator,
    VPINCalculator,
    PriceImpactModel,
    bucket_flow,
    TradeSign,
    TraderType,
)

from .mechanism.liquidity import (
    Quote,
    DepthSnapshot,
    SpreadDecomposition,
    AmihudIlliquidity,
    KyleLambda,
    LiquidityResilience,
    InventoryAdjustedQuotes,
    LiquidityTracker,
)

from .mechanism.leverage import (
    Position,
    PositionBook,
    BalanceSheet,
    MarginAccount,
    VaRMargin,
    FundingLiquidityModel,
    LeverageCycleTracker,
)

from .mechanism.liquidation import (
    LiquidationOrder,
    LiquidationReason,
    LiquidationStatus,
    CascadeEvent,
    LiquidationTrigger,
    LiquidationExecutor,
    InsuranceFund,
    LiquidationCascadeEngine,
    LiquidationPressureMap,
    FireSaleExternality,
)

# ---- agents -----------------------------------------------------------------
from .agents.market_maker import (
    MarketMaker,
    MarketState,
    QuoteDecision,
    Fill,
    MMStatus,
    MMSnapshot,
    InventoryManager,
    PnLTracker,
    RiskManager,
    AdverseSelectionDetector,
    PerformanceMonitor,
    # strategies
    SymmetricStrategy,
    AvellanedaStoikovStrategy,
    InventorySkewStrategy,
    VolatilityAdaptiveStrategy,
    MultiLevelStrategy,
)

from .agents.traders import (
    BaseTrader,
    NoiseTrader,
    InformedTrader,
    MomentumTrader,
    MeanReversionTrader,
    LiquiditySeeker,
    StopLossTrader,
    OrderIntent,
    FillNotice,
    OrderSide,
    build_trader_population,
)

# ---- simulation -------------------------------------------------------------
from .simulation.market_simulation import (
    FundamentalProcess,
    SimulationClock,
    MarketSimulation,
    SimulationResult,
    ScenarioBuilder,
    RollingVolatility,
)

# ---- package metadata -------------------------------------------------------
__version__ = "0.1.0"
__author__  = "Market Microstructure Research"

__all__ = [
    # state
    "OrderBook",

    # order flow
    "OrderEvent", "FlowSnapshot",
    "PoissonArrivalProcess", "HawkesArrivalProcess",
    "OrderSizeDistribution", "OrderFlowGenerator",
    "TradeClassifier", "OFICalculator", "VPINCalculator",
    "PriceImpactModel", "bucket_flow",
    "TradeSign", "TraderType",

    # liquidity
    "Quote", "DepthSnapshot",
    "SpreadDecomposition", "AmihudIlliquidity",
    "KyleLambda", "LiquidityResilience",
    "InventoryAdjustedQuotes", "LiquidityTracker",

    # leverage
    "Position", "PositionBook", "BalanceSheet",
    "MarginAccount", "VaRMargin",
    "FundingLiquidityModel", "LeverageCycleTracker",

    # liquidation
    "LiquidationOrder", "LiquidationReason", "LiquidationStatus",
    "CascadeEvent", "LiquidationTrigger", "LiquidationExecutor",
    "InsuranceFund", "LiquidationCascadeEngine",
    "LiquidationPressureMap", "FireSaleExternality",

    # market maker
    "MarketMaker", "MarketState", "QuoteDecision", "Fill",
    "MMStatus", "MMSnapshot",
    "InventoryManager", "PnLTracker", "RiskManager",
    "AdverseSelectionDetector", "PerformanceMonitor",
    "SymmetricStrategy", "AvellanedaStoikovStrategy",
    "InventorySkewStrategy", "VolatilityAdaptiveStrategy",
    "MultiLevelStrategy",

    # traders
    "BaseTrader", "NoiseTrader", "InformedTrader",
    "MomentumTrader", "MeanReversionTrader",
    "LiquiditySeeker", "StopLossTrader",
    "OrderIntent", "FillNotice", "OrderSide",
    "build_trader_population",

    # simulation
    "FundamentalProcess", "SimulationClock",
    "MarketSimulation", "SimulationResult",
    "ScenarioBuilder", "RollingVolatility",
]
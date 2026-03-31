# Monitor Subsystem

## Responsibility
Live tracking of portfolio performance, system health, and execution status. Reporting.

## Inputs
- Live market data.
- Broker account status.

## Outputs
- Dashboards.
- Alerts.
- End-of-Day (EOD) Reports.

## Key Components
- `accountActivity.py`: Tracks account history.
- `portfolio_performance.py`: Calculates and reports portfolio metrics.
- `reports/`: Directory storing generated reports.

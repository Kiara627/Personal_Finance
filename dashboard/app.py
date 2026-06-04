"""
Personal Finance Analytics Dashboard
Built with Plotly Dash · Data: personal finance data.csv
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── Load & Preprocess ─────────────────────────────────────────────────────────
df = pd.read_csv("../personal finance data.csv")
df["Date"] = pd.to_datetime(df["Date / Time"], format="%d %B %Y")
df["Month"] = df["Date"].dt.to_period("M")
df["Category"] = df["Category"].str.strip().str.title()

income_df = df[df["Income/Expense"] == "Income"]
expense_df = df[df["Income/Expense"] == "Expense"]

monthly_income = income_df.groupby("Month")["Debit/Credit"].sum()
monthly_expense = expense_df.groupby("Month")["Debit/Credit"].sum()
monthly_savings = monthly_income - monthly_expense

monthly = pd.DataFrame(
    {"Income": monthly_income, "Expense": monthly_expense, "Savings": monthly_savings}
).reset_index()
monthly["Month_dt"] = monthly["Month"].dt.to_timestamp()

# ── KPI Metrics ───────────────────────────────────────────────────────────────
total_income = monthly_income.sum()
total_expenses = monthly_expense.sum()
total_savings = monthly_savings.sum()
avg_savings_rate = (monthly_savings / monthly_income * 100).mean()

# ── ARIMA Forecast ────────────────────────────────────────────────────────────
savings_ts = monthly_savings.copy()
savings_ts.index = savings_ts.index.to_timestamp()
try:
    savings_ts = savings_ts.asfreq("MS")
    arima_model = ARIMA(savings_ts, order=(1, 1, 1))
    arima_result = arima_model.fit()
    arima_forecast = arima_result.forecast(steps=3)
except Exception:
    arima_forecast = pd.Series(
        [savings_ts.mean()] * 3,
        index=pd.date_range(savings_ts.index[-1], periods=4, freq="MS")[1:],
    )

forecast_dates = pd.date_range(savings_ts.index[-1], periods=4, freq="MS")[1:]

# ── K-Means Clustering ────────────────────────────────────────────────────────
cluster_df = (
    expense_df.groupby("Category")["Debit/Credit"].sum().reset_index()
)
scaler = StandardScaler()
cluster_df["Scaled"] = scaler.fit_transform(cluster_df[["Debit/Credit"]])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_df["_cluster"] = kmeans.fit_predict(cluster_df[["Scaled"]])

# Map cluster IDs to meaningful labels based on average spend per cluster
mean_by_cluster = cluster_df.groupby("_cluster")["Debit/Credit"].mean().sort_values()
label_map = dict(
    zip(mean_by_cluster.index, ["Low Spend", "Medium Spend", "High Spend"])
)
cluster_df["Segment"] = cluster_df["_cluster"].map(label_map)

# ── Color Palette ─────────────────────────────────────────────────────────────
C = {
    "bg":       "#0F1117",
    "card":     "#1A1D27",
    "border":   "#2A2D3A",
    "text":     "#E2E8F0",
    "subtext":  "#64748B",
    "income":   "#10B981",
    "expense":  "#F43F5E",
    "savings":  "#38BDF8",
    "forecast": "#FBBF24",
}

CLUSTER_COLORS = {"Low Spend": "#38BDF8", "Medium Spend": "#FBBF24", "High Spend": "#F43F5E"}

CHART_LAYOUT = dict(
    plot_bgcolor=C["card"],
    paper_bgcolor=C["card"],
    font=dict(color=C["text"], family="Inter, system-ui, sans-serif"),
    margin=dict(l=50, r=30, t=50, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    xaxis=dict(gridcolor=C["border"], zeroline=False),
    yaxis=dict(gridcolor=C["border"], zeroline=False),
)

# ── Chart Builders ────────────────────────────────────────────────────────────
def monthly_trends_chart():
    fig = go.Figure()
    for col, color, name in [
        ("Income",  C["income"],   "Income"),
        ("Expense", C["expense"],  "Expenses"),
        ("Savings", C["savings"],  "Savings"),
    ]:
        fig.add_trace(go.Scatter(
            x=monthly["Month_dt"],
            y=monthly[col],
            name=name,
            mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=7, line=dict(color=C["bg"], width=1.5)),
            hovertemplate=f"<b>{name}</b>: ₹%{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text="Monthly Income, Expenses & Savings", font=dict(size=14)),
        xaxis_title="Month",
        yaxis_title="Amount (₹)",
        hovermode="x unified",
        **CHART_LAYOUT,
    )
    return fig


def category_spend_chart():
    cat = (
        expense_df.groupby("Category")["Debit/Credit"]
        .sum()
        .sort_values(ascending=True)
    )
    bar_colors = px.colors.sample_colorscale(
        "Teal", [i / (len(cat) - 1) for i in range(len(cat))]
    )
    fig = go.Figure(go.Bar(
        x=cat.values,
        y=cat.index,
        orientation="h",
        marker=dict(color=bar_colors),
        text=[f"₹{v:,.0f}" for v in cat.values],
        textposition="outside",
        textfont=dict(size=11, color=C["text"]),
        hovertemplate="<b>%{y}</b>: ₹%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Spending by Category", font=dict(size=14)),
        xaxis_title="Total Expense (₹)",
        xaxis=dict(gridcolor=C["border"], zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", zeroline=False),
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        margin=dict(l=120, r=90, t=50, b=50),
    )
    return fig


def arima_forecast_chart():
    fig = go.Figure()
    # Historical
    fig.add_trace(go.Scatter(
        x=savings_ts.index,
        y=savings_ts.values,
        name="Historical Savings",
        mode="lines+markers",
        line=dict(color=C["savings"], width=2.5),
        marker=dict(size=7, line=dict(color=C["bg"], width=1.5)),
        hovertemplate="<b>Historical</b>: ₹%{y:,.0f}<extra></extra>",
    ))
    # Connector dotted line
    fig.add_trace(go.Scatter(
        x=[savings_ts.index[-1], forecast_dates[0]],
        y=[savings_ts.values[-1], arima_forecast.values[0]],
        mode="lines",
        line=dict(color=C["forecast"], dash="dot", width=1.5),
        showlegend=False,
        hoverinfo="skip",
    ))
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=arima_forecast.values,
        name="ARIMA Forecast",
        mode="lines+markers",
        line=dict(color=C["forecast"], width=2.5, dash="dash"),
        marker=dict(size=9, symbol="diamond", line=dict(color=C["bg"], width=1.5)),
        hovertemplate="<b>Forecast</b>: ₹%{y:,.0f}<extra></extra>",
    ))
    # Forecast region shading
    fig.add_vrect(
        x0=str(savings_ts.index[-1]),
        x1=str(forecast_dates[-1]),
        fillcolor=C["forecast"],
        opacity=0.05,
        line_width=0,
    )
    fig.update_layout(
        title=dict(text="ARIMA(1,1,1) Savings Forecast — Next 3 Months", font=dict(size=14)),
        xaxis_title="Month",
        yaxis_title="Savings (₹)",
        hovermode="x unified",
        **CHART_LAYOUT,
    )
    return fig


def kmeans_cluster_chart():
    cluster_df_sorted = cluster_df.sort_values("Debit/Credit")
    fig = go.Figure()
    for segment, color in CLUSTER_COLORS.items():
        subset = cluster_df_sorted[cluster_df_sorted["Segment"] == segment]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset["Category"],
            y=subset["Debit/Credit"],
            mode="markers+text",
            name=segment,
            text=subset["Category"],
            textposition="top center",
            textfont=dict(size=11),
            marker=dict(
                size=subset["Debit/Credit"] / subset["Debit/Credit"].max() * 50 + 18,
                color=color,
                opacity=0.85,
                line=dict(color=C["bg"], width=2),
            ),
            hovertemplate="<b>%{x}</b><br>Total: ₹%{y:,.0f}<br>Segment: " + segment + "<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text="K-Means Spending Segments (k=3)", font=dict(size=14)),
        xaxis_title="Expense Category",
        yaxis_title="Total Spend (₹)",
        showlegend=True,
        **CHART_LAYOUT,
        margin=dict(l=50, r=30, t=60, b=60),
    )
    return fig


# ── KPI Card Component ────────────────────────────────────────────────────────
def kpi_card(label, value, accent_color, icon=""):
    return dbc.Card(
        dbc.CardBody([
            html.P(
                f"{icon}  {label}" if icon else label,
                style={
                    "color": C["subtext"],
                    "fontSize": "0.72rem",
                    "textTransform": "uppercase",
                    "letterSpacing": "1.2px",
                    "margin": "0 0 6px 0",
                    "fontWeight": "600",
                },
            ),
            html.H3(
                value,
                style={
                    "color": accent_color,
                    "fontWeight": "700",
                    "margin": 0,
                    "fontSize": "1.6rem",
                },
            ),
        ]),
        style={
            "backgroundColor": C["card"],
            "border": f"1px solid {accent_color}40",
            "borderRadius": "12px",
            "boxShadow": f"0 0 20px {accent_color}15",
        },
    )


# ── App Layout ────────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap",
    ],
)
app.title = "Personal Finance Analytics"

GRAPH_CFG = {"displayModeBar": False, "responsive": True}

app.layout = dbc.Container(
    [
        # ── Header ──────────────────────────────────────────────────────────
        html.Div(
            [
                html.H2(
                    "Personal Finance Analytics",
                    style={"color": C["text"], "fontWeight": "700", "margin": 0, "fontSize": "1.6rem"},
                ),
                html.P(
                    "2021 Transaction Data  ·  ARIMA Forecasting  ·  K-Means Segmentation",
                    style={"color": C["subtext"], "margin": "4px 0 0 0", "fontSize": "0.85rem"},
                ),
            ],
            style={"padding": "28px 0 20px 0", "borderBottom": f"1px solid {C['border']}"},
        ),

        html.Div(style={"height": "20px"}),

        # ── KPI Cards ────────────────────────────────────────────────────────
        dbc.Row(
            [
                dbc.Col(kpi_card("Total Income",              f"₹{total_income:,.0f}",     C["income"],   "↑"), md=3),
                dbc.Col(kpi_card("Total Expenses",            f"₹{total_expenses:,.0f}",   C["expense"],  "↓"), md=3),
                dbc.Col(kpi_card("Net Savings",               f"₹{total_savings:,.0f}",    C["savings"],  "◆"), md=3),
                dbc.Col(kpi_card("Avg Monthly Savings Rate",  f"{avg_savings_rate:.1f}%",  C["forecast"], "◎"), md=3),
            ],
            className="g-3",
        ),

        html.Div(style={"height": "24px"}),

        # ── Charts Row 1 ─────────────────────────────────────────────────────
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dcc.Graph(figure=monthly_trends_chart(), config=GRAPH_CFG, style={"height": "360px"}),
                        style={"backgroundColor": C["card"], "border": f"1px solid {C['border']}", "borderRadius": "12px"},
                    ),
                    md=8,
                ),
                dbc.Col(
                    dbc.Card(
                        dcc.Graph(figure=category_spend_chart(), config=GRAPH_CFG, style={"height": "360px"}),
                        style={"backgroundColor": C["card"], "border": f"1px solid {C['border']}", "borderRadius": "12px"},
                    ),
                    md=4,
                ),
            ],
            className="g-3",
        ),

        html.Div(style={"height": "20px"}),

        # ── Charts Row 2 ─────────────────────────────────────────────────────
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dcc.Graph(figure=arima_forecast_chart(), config=GRAPH_CFG, style={"height": "360px"}),
                        style={"backgroundColor": C["card"], "border": f"1px solid {C['border']}", "borderRadius": "12px"},
                    ),
                    md=6,
                ),
                dbc.Col(
                    dbc.Card(
                        dcc.Graph(figure=kmeans_cluster_chart(), config=GRAPH_CFG, style={"height": "360px"}),
                        style={"backgroundColor": C["card"], "border": f"1px solid {C['border']}", "borderRadius": "12px"},
                    ),
                    md=6,
                ),
            ],
            className="g-3",
        ),

        # ── Footer ────────────────────────────────────────────────────────────
        html.P(
            "Built with Plotly Dash  ·  Data: personal finance data.csv  ·  2021",
            style={
                "color": C["subtext"],
                "fontSize": "0.72rem",
                "textAlign": "center",
                "padding": "24px 0 16px 0",
                "borderTop": f"1px solid {C['border']}",
                "marginTop": "20px",
            },
        ),
    ],
    fluid=True,
    style={"backgroundColor": C["bg"], "minHeight": "100vh", "fontFamily": "Inter, system-ui, sans-serif"},
)

if __name__ == "__main__":
    app.run(debug=True)

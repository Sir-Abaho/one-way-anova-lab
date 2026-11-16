# One-Way ANOVA Lab

A standalone Streamlit application that calculates one-way ANOVA from its
definitions, checks the results against SciPy, and reports assumptions,
Tukey HSD follow-ups, effect size, and plots.

## Run

On this computer, double-click `run_anova.bat` to start the app without
entering a command. Keep the console window open while using the app; press
Ctrl+C in that window to stop it.

Alternatively, open a terminal in this folder and run:

```text
pip install -r requirements.txt
streamlit run anova_app.py
```

Run the worked-example tests with:

```text
python -m unittest -v test_anova_app.py
```

## Input data

Choose **Enter values** to enter 2–6 named groups as comma-separated finite
numbers, or **Upload CSV** to load a CSV containing one group per column.
Blank CSV cells are ignored, so groups may have different sample sizes. Each
group must contain at least two observations.

The explanation document is `anova_explanation.pdf`.

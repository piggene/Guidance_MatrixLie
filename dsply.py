import plotly.io as pio


pio.from_json(open('test_results/equigrasp_guidance/20251111-1556_none/visualizations.json', 'r').read()).show(renderer="browser") 

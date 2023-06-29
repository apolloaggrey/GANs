from bokeh.plotting import figure, show, curdoc
from bokeh.models import ColumnDataSource
from bokeh.layouts import column
from bokeh.driving import linear
import pandas as pd

# Load data from CSV file
data = pd.read_csv('losses.csv')

# Create ColumnDataSource from data
source = ColumnDataSource(data)

# Create line chart for generator loss
gen_loss_plot = figure(width=600, height=300, title='Generator Loss')
gen_loss_plot.line(x='epoch', y='gen_loss', source=source)

# Create line chart for discriminator loss
disc_loss_plot = figure(width=600, height=300, title='Discriminator Loss')
disc_loss_plot.line(x='epoch', y='disc_loss', source=source)

# Combine plots vertically
plot = column(gen_loss_plot, disc_loss_plot)

# Update function to update data source and plot
@linear()
def update(step):
    # Load new data from CSV file
    new_data = pd.read_csv('losses.csv')
    # Update data source with new data
    source.data = ColumnDataSource.from_df(new_data)

# Add update function to document
curdoc().add_root(plot)
curdoc().add_periodic_callback(update, 1000)
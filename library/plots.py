from plotly import __version__
from plotly.offline import (download_plotlyjs, plot)
from plotly.graph_objs import (Bar, Scatter, Figure,
                               Pie, Layout, Line, Annotations, Annotation)
from plotly import tools
from plotly.tools import FigureFactory as FF


class ScatterPlot(object):
    def __init__(self, x, datatitle, xtitle, y, ytitle, plottitle):
        """
        Creates a scatter plot from the data.
        
        :param x:  X values, as a list. 
        :param datatitle:  Title for the data in the legend.
        :param xtitle:  Title for the X axis.
        :param y:  Y values, as a list.
        :param ytitle:  Title for the Y axis.
        :param plottitle:  Title for the overall plot. 
        """
        self._output = []

        # Add current scatter plot
        self._output.append(Scatter(name=datatitle,
                                   x=x,
                                   y=y,
                                   hoverinfo="x+y"
                                   ))

        self._plotlayout = Layout(showlegend=True, title=plottitle,
                            xaxis=dict(title=xtitle),
                            yaxis=dict(title=ytitle))
        self._plotfigure = Figure(data=self._output, layout=self._plotlayout)

    def plot_div(self):
        """
        Returns the plot as an HTML div for embedding.
        
        :return:  The HTML div of the plot in this object. 
        """
        # Plot without the plotly annoying link...
        return plot(self._plotfigure, show_link=False, output_type='div',
             auto_open=False)

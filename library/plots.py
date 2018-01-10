from plotly import __version__
from plotly.offline import (download_plotlyjs, plot)
from plotly.graph_objs import (Bar, Scatter, Figure,
                               Pie, Layout, Line, Annotations, Annotation)
from plotly import tools
from plotly.tools import FigureFactory as FF


class ScatterPlot(object):
    def __init__(self, x, datatitle, xtitle, y, ytitle, plottitle, mode=None, text=None):
        """
        Creates a scatter plot from the data.
        
        :param x:  X axis values, as a list.  It is a list of lists.
        :param datatitle:  A list of titles for the data in the legend for each Y axis.
        :param xtitle:  Title for the X axis.
        :param y:  Y axis values, as a list.  It is a list of lists.
        :param ytitle:  Title for the Y axis.
        :param plottitle:  Title for the overall plot.
        :param mode: A list of modes for the scatter plot, None is line
        :param text: A list of dicts that has text, x, and y for the text annotation.
        """
        self._output = []

        # Add for each y in the multi plot
        for i in range(0, len(y)):
            # Get the mode if available
            if mode is None:
                mymode = 'line'
            else:
                mymode = mode[i]
            # Add current scatter plot
            self._output.append(Scatter(name=datatitle[i],
                                        x=x[i],
                                        y=y[i],
                                        hoverinfo="x+y",
                                        mode=mymode
                                       ))

        annotations = list()

        # Add text annotations
        if text:
            for t in text:
                # Add annotations
                annotations.append(
                    dict(
                        x=t['x'],
                        y=t['y'],
                        xref='x',
                        yref='y',
                        text=t['text'],
                        showarrow=True
                    )
                )

        # # Add text annotations
        # if text:
        #     for t in text:
        #         mymode = 'text'
        #         # Add annotations
        #         self._output.append(Scatter(text=[t['text']],
        #                                     x=[t['x']],
        #                                     y=[t['y']],
        #                                     hoverinfo="x",
        #                                     mode=mymode,
        #                                     textposition='right'
        #                                    ))

        self._plotlayout = Layout(showlegend=True, title=plottitle,
                                  xaxis=dict(title=xtitle),
                                  yaxis=dict(title=ytitle),
                                  annotations=annotations)
        self._plotfigure = Figure(data=self._output, layout=self._plotlayout)

    def plot_div(self):
        """
        Returns the plot as an HTML div for embedding.
        
        :return:  The HTML div of the plot in this object. 
        """
        # Plot without the plotly annoying link...
        return plot(self._plotfigure, show_link=False, output_type='div',
             auto_open=False)

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as widgets

from itertools import cycle
from jupyter_jsmol import JsmolView
from IPython.display import display, HTML, FileLink


class Visualizer:

    def __init__(self, df, df_grouped, hover_features, color_features):

        self.marker_size = 7
        self.symbols = [
            'circle',
            'square',
            'triangle-up',
            'triangle-down',
            'circle-cross',
            'circle-x'
        ]
        self.font_size = 12
        self.cross_size = 20
        self.line_width = 1
        self.font_families = ['Source Sans Pro',
                              'Helvetica',
                              'Open Sans',
                              'Times New Roman',
                              'Arial',
                              'Verdana',
                              'Courier New',
                              'Comic Sans MS',
                              ]
        self.line_styles = ["dash",
                            "solid",
                            "dot",
                            "longdash",
                            "dashdot",
                            "longdashdot"]
        self.qualitative_colors = ['Plotly',
                                   'D3',
                                   'G10',
                                   'T10',
                                   'Alphabet',
                                   'Dark24',
                                   'Light24',
                                   'Set1',
                                   'Pastel1',
                                   'Dark2',
                                   'Set2',
                                   'Pastel2',
                                   'Set3',
                                   'Antique',
                                   'Bold',
                                   'Pastel',
                                   'Prism',
                                   'Safe',
                                   'Vivid'
                                   ]
        self.bg_color = 'rgba(229,236,246, 0.5)'
        self.bg_toggle = True
        self.hover_features = hover_features
        self.color_features = ['Clustering'] + color_features
        self.df = df
        self.df_grouped = df_grouped
        for col in self.df_grouped.columns:
            if self.df_grouped[col].dtype == 'float64':
                self.df_grouped[col] = self.df_grouped[col].to_numpy().round(decimals=4)

        self.replica_l = 0
        self.replica_r = 0
        self.trace_l = []
        self.trace_r = []
        self.n_clusters = self.df_grouped['Cluster_label'].max() + 1
        self.global_symbols = []
        self.global_sizes = []
        self.global_markerlinecolor = []
        self.global_markerlinewidth = []
        self.n_points = []
        self.total_compounds = df_grouped.shape[0]
        self.frac = (1000 / self.total_compounds)
        if self.frac > 1:
            self.frac = 1
        self.frac = int(self.frac*100)/100
        self.fig = go.FigureWidget()
        self.viewer_l = JsmolView()
        self.viewer_r = JsmolView()
        self.instantiate_widgets()
        self.palette = cycle(getattr(px.colors.qualitative, self.qualitative_colors[0]))
        self.hover_text = []
        self.hover_custom = []
        self.hover_template = []
        self.shuffled_entries = []
        self.df_clusters = []
        self.df_entries_onmap = []
        self.name_trace = []
        self.trace = {}

        for cl in np.concatenate([np.arange(self.n_clusters), np.array([-1])]):
            self.df_clusters.append(self.df_grouped.loc[self.df_grouped['Cluster_label'] == cl])
            self.shuffled_entries.append(self.df_clusters[cl].index.to_numpy()[
                                            np.random.permutation(self.df_clusters[cl].shape[0])])
            self.n_points.append(int(self.frac * self.df_clusters[cl].shape[0]))
            self.global_symbols.append(["circle"] * self.n_points[cl])
            self.global_sizes.append([self.marker_size] * self.n_points[cl])
            self.global_markerlinecolor.append(['white'] * self.n_points[cl])
            self.global_markerlinewidth.append([1] * self.n_points[cl])
            self.df_entries_onmap.append(self.df_clusters[cl].loc[self.shuffled_entries[cl]].head(self.n_points[cl]))

        for cl in range(self.n_clusters):
            self.name_trace.append('Cluster ' + str(cl))
            self.fig.add_trace(
                (
                    go.Scatter(
                        name=self.name_trace[cl],
                        mode='markers',
                        x=self.df_entries_onmap[cl]['x_emb'],
                        y=self.df_entries_onmap[cl]['y_emb'],
                        marker_color=next(self.palette),
                    )))
            self.trace[self.name_trace[cl]] = self.fig['data'][cl]
        self.name_trace.append('Outliers')
        self.fig.add_trace(
            (
                go.Scatter(
                    name=self.name_trace[self.n_clusters],
                    mode='markers',
                    x=self.df_entries_onmap[self.n_clusters]['x_emb'],
                    y=self.df_entries_onmap[self.n_clusters]['y_emb'],
                    marker_color=next(self.palette),
                    visible='legendonly'
                )))
        self.trace[self.name_trace[self.n_clusters]] = self.fig['data'][self.n_clusters]
        self.name_trace.append('All')
        self.df_entries_onmap.append(pd.concat(self.df_entries_onmap[:self.n_clusters + 1]))
        self.n_points.append(int(self.df_entries_onmap[-1].shape[0]))
        self.global_symbols.append(["circle"] * self.n_points[-1])
        self.global_sizes.append([self.marker_size] * self.n_points[-1])
        self.global_markerlinecolor.append(['white'] * self.n_points[-1])
        self.global_markerlinewidth.append([1] * self.n_points[-1])
        self.fig.add_trace(
            (
                go.Scatter(
                    name=self.name_trace[-1],
                    mode='markers',
                    x=self.df_entries_onmap[-1]['x_emb'],
                    y=self.df_entries_onmap[-1]['y_emb'],
                    visible=False
                )))
        self.trace[self.name_trace[-1]] = self.fig['data'][-1]

        self.fig.update_xaxes(ticks="outside", tickwidth=1, ticklen=10, linewidth=1, linecolor='black')
        self.fig.update_yaxes(ticks="outside", tickwidth=1, ticklen=10, linewidth=1, linecolor='black')
        x_min = min(self.df_grouped['x_emb'])
        y_min = min(self.df_grouped['y_emb'])
        x_max = max(self.df_grouped['x_emb'])
        y_max = max(self.df_grouped['y_emb'])
        x_delta = 0.05 * abs(x_max - x_min)
        y_delta = 0.05 * abs(y_max - y_min)
        self.fig.update_layout(
            xaxis_title="x_emb",
            yaxis_title="y_emb",
            xaxis_range=[x_min - x_delta, x_max + x_delta],
            yaxis_range=[y_min - y_delta, y_max + y_delta],
            hoverlabel=dict(
                bgcolor="white",
                font_size=16,
                font_family="Rockwell"
            ),
            width=800,
            height=400,
            margin=dict(l=50, r=50, b=70, t=20, pad=4),
            legend_itemsizing='constant'
        )

        self.update_appearance_variables()
        self.update_layout_figure()
        self.fig.update_traces(
            selector={'name': 'Outliers'},
            visible='legendonly'
        )

    def update_layout_figure(self):

        with self.fig.batch_update():

            if self.widg_colormarkers.value == 'Clustering':
                self.palette = cycle(getattr(px.colors.qualitative, self.widg_colorpalette.value))
                self.fig.update_layout(showlegend=True)
                for cl in np.arange(self.n_clusters+1):
                    color = next(self.palette)
                    self.trace[self.name_trace[cl]].marker.symbol = self.global_symbols[cl]
                    self.trace[self.name_trace[cl]].marker.size = self.global_sizes[cl]
                    self.trace[self.name_trace[cl]].marker.line.color = self.global_markerlinecolor[cl]
                    self.trace[self.name_trace[cl]].marker.line.width = self.global_markerlinewidth[cl]
                    self.trace[self.name_trace[cl]]['x'] = self.df_entries_onmap[cl]['x_emb']
                    self.trace[self.name_trace[cl]]['y'] = self.df_entries_onmap[cl]['y_emb']
                    self.fig.update_traces(
                        selector={'name': self.name_trace[cl]},
                        text=self.hover_text[cl],
                        customdata=self.hover_custom[cl],
                        hovertemplate=self.hover_template[cl],
                        marker_color=color,
                        visible=True
                    )
                self.trace[self.name_trace[-1]].marker.symbol = []
                self.trace[self.name_trace[-1]].marker.size = []
                self.trace[self.name_trace[-1]].marker.line.color = []
                self.trace[self.name_trace[-1]].marker.line.width = []
                self.trace[self.name_trace[-1]]['x'] = []
                self.trace[self.name_trace[-1]]['y'] = []
                self.fig.update_traces(
                    selector={'name': self.name_trace[-1]},
                    text=self.hover_text[-1],
                    customdata=self.hover_custom[-1],
                    hovertemplate=self.hover_template[-1],
                    visible=False
                )
            else:
                self.fig.update_layout(showlegend=False)
                for cl in np.arange(self.n_clusters+1):
                    self.trace[self.name_trace[cl]].marker.symbol = []
                    self.trace[self.name_trace[cl]].marker.size = []
                    self.trace[self.name_trace[cl]].marker.line.color = []
                    self.trace[self.name_trace[cl]].marker.line.width = []
                    self.trace[self.name_trace[cl]]['x'] = []
                    self.trace[self.name_trace[cl]]['y'] = []
                    self.fig.update_traces(
                        selector={'name': self.name_trace[cl]},
                        text=self.hover_text[cl],
                        customdata=self.hover_custom[cl],
                        hovertemplate=self.hover_template[cl],
                        visible=False
                    )
                self.trace[self.name_trace[-1]].marker.symbol = self.global_symbols[-1]
                self.trace[self.name_trace[-1]].marker.size = self.global_sizes[-1]
                self.trace[self.name_trace[-1]].marker.line.color = self.global_markerlinecolor[-1]
                self.trace[self.name_trace[-1]].marker.line.width = self.global_markerlinewidth[-1]
                self.trace[self.name_trace[-1]]['x'] = self.df_entries_onmap[-1]['x_emb']
                self.trace[self.name_trace[-1]]['y'] = self.df_entries_onmap[-1]['y_emb']
                color = self.df_entries_onmap[-1][self.widg_colormarkers.value]
                self.fig.update_traces(
                    selector={'name': self.name_trace[-1]},
                    marker=dict(color=color, colorscale=self.widg_continuouscolors.value),
                    text=self.hover_text[-1],
                    customdata=self.hover_custom[-1],
                    hovertemplate=self.hover_template[-1],
                    visible=True,
                )

    def make_dfclusters(self):

        if self.trace_l:
            trace_l, formula_l = self.trace_l
        else:
            trace_l = -2
        if self.trace_r:
            trace_r, formula_r = self.trace_r
        else:
            trace_r = -2

        for cl in range(self.n_clusters+1):
            self.df_entries_onmap[cl] = self.df_clusters[cl].loc[self.shuffled_entries[cl]].head(int(self.frac *
                                                                                                     self.df_clusters[cl].shape[0]))
            if cl == trace_l:
                self.df_entries_onmap[cl] = pd.concat([
                    self.df_entries_onmap[cl],
                    self.df_clusters[trace_l].loc[[formula_l]]
                ], axis=0)
            if cl == trace_r:
                self.df_entries_onmap[cl] = pd.concat([
                    self.df_entries_onmap[cl],
                    self.df_clusters[trace_r].loc[[formula_r]],
                ], axis=0)
            self.n_points[cl] = self.df_entries_onmap[cl].shape[0]

        self.reset_markers()
        for cl in range(self.n_clusters+1):
            try:
                try:
                    point = np.where(self.df_entries_onmap[cl].index.to_numpy() == formula_l)[0][1]
                    self.global_symbols[cl][point] = 'x'
                except:
                    point = np.where(self.df_entries_onmap[cl].index.to_numpy() == formula_l)[0][0]
                    self.global_symbols[cl][point] = 'x'
            except:
                pass
            try:
                try:
                    point = np.where(self.df_entries_onmap[cl].index.to_numpy() == formula_r)[0][1]
                    self.global_symbols[cl][point] = 'cross'
                except:
                    point = np.where(self.df_entries_onmap[cl].index.to_numpy() == formula_r)[0][0]
                    self.global_symbols[cl][point] = 'cross'
            except:
                pass

        if self.widg_outliersbox.value:
            self.df_entries_onmap[-1] = pd.concat(self.df_entries_onmap[:self.n_clusters + 1], axis=0, sort=False)
            self.n_points[-1] = int(self.df_entries_onmap[-1].shape[0])
            self.global_symbols[-1] = [symb for sub in self.global_symbols[:-1] for symb in sub]
        else:
            self.df_entries_onmap[-1] = pd.concat(self.df_entries_onmap[:self.n_clusters], axis=0, sort=False)
            self.n_points[-1] = int(self.df_entries_onmap[-1].shape[0])
            self.global_symbols[-1] = [symb for sub in self.global_symbols[:-2] for symb in sub]

    def reset_markers(self):

        for cl in range(self.n_clusters + 2):
            self.global_symbols[cl] = ["circle"] * (self.n_points[cl])
            self.global_sizes[cl] = [self.marker_size] * self.n_points[cl]
            self.global_markerlinecolor[cl] = ['white'] * self.n_points[cl]
            self.global_markerlinewidth[cl] = [1] * self.n_points[cl]

    def update_appearance_variables(self):

        self.hover_text = []
        self.hover_custom = []
        self.hover_template = []
        for cl in range(self.n_clusters+1):
            self.hover_text.append(self.df_entries_onmap[cl].index)
            hover_template = r"<b>%{text}</b><br><br>"
            if self.hover_features:
                hover_custom = np.dstack([self.df_entries_onmap[cl][self.hover_features[0]].to_numpy()])
                hover_template += str(self.hover_features[0]) + ": %{customdata[0]}<br>"
                for i in range(1, len(self.hover_features), 1):
                    hover_custom = np.dstack([hover_custom, self.df_entries_onmap[cl][self.hover_features[i]].to_numpy()])
                    hover_template += str(self.hover_features[i]) + ": %{customdata[" + str(i) + "]}<br>"
                self.hover_custom.append(hover_custom[0])
                self.hover_template.append(hover_template)
            else:
                self.hover_custom.append([''])
                self.hover_template.append([''])
        self.hover_text.append(self.df_entries_onmap[-1].index)
        hover_template = r"<b>%{text}</b><br><br>"
        if self.hover_features:
            hover_custom = np.dstack([self.df_entries_onmap[-1][self.hover_features[0]].to_numpy()])
            hover_template += str(self.hover_features[0]) + ": %{customdata[0]}<br>"
            for i in range(1, len(self.hover_features), 1):
                hover_custom = np.dstack([hover_custom, self.df_entries_onmap[-1][self.hover_features[i]].to_numpy()])
                hover_template += str(self.hover_features[i]) + ": %{customdata[" + str(i) + "]}<br>"
            self.hover_custom.append(hover_custom[0])
            self.hover_template.append(hover_template)
        else:
            self.hover_custom.append([''])
            self.hover_template.append([''])

        for cl in np.arange(self.n_clusters+1):
            markerlinewidth = [1] * self.n_points[cl]
            markerlinecolor = ['white'] * self.n_points[cl]
            sizes = [self.marker_size] * self.n_points[cl]
            symbols = self.global_symbols[cl]
            try:
                point = symbols.index('x')
                sizes[point] = self.cross_size
                markerlinewidth[point] = 2
                markerlinecolor[point] = 'black'
            except:
                pass
            try:
                point = symbols.index('cross')
                sizes[point] = self.cross_size
                markerlinewidth[point] = 2
                markerlinecolor[point] = 'black'
            except:
                pass
            self.global_sizes[cl] = sizes
            self.global_markerlinecolor[cl] = markerlinecolor
            self.global_markerlinewidth[cl] = markerlinewidth
        self.global_sizes[-1] = [symb for sub in self.global_sizes[:-1] for symb in sub]
        self.global_markerlinecolor[-1] = [symb for sub in self.global_markerlinecolor[:-1] for symb in sub]
        self.global_markerlinewidth[-1] = [symb for sub in self.global_markerlinewidth[:-1] for symb in sub]

    def display_button_l_clicked(self, button):

        # Actions are performed only if the string inserted in the text widget corresponds to an existing compound
        if self.widg_compound_text_l.value in self.df_grouped.index.tolist():

            self.replica_l += 1
            formula_l = self.widg_compound_text_l.value
            # self.view_structure_l(formula_l)

            trace_l = self.df_grouped.loc[self.df_grouped.index == formula_l]['Cluster_label'][0]
            if trace_l == -1:
                trace_l = self.n_clusters

            self.trace_l = [trace_l, formula_l]

            self.make_dfclusters()
            self.update_appearance_variables()
            self.update_layout_figure()

            if self.widg_colormarkers.value == 'Clustering':
                name_trace = self.name_trace[trace_l]
                with self.fig.batch_update():
                    self.fig.update_traces(
                        selector={'name': name_trace},
                        visible=True
                    )

    def display_button_r_clicked(self, button):

        # Actions are performed only if the string inserted in the text widget corresponds to an existing compound
        if self.widg_compound_text_r.value in self.df_grouped.index.tolist():

            self.replica_r += 1
            formula_r = self.widg_compound_text_r.value
            # self.view_structure_r(formula_r)

            trace_r = self.df_grouped.loc[self.df_grouped.index == formula_r]['Cluster_label'][0]
            if trace_r == -1:
                trace_r = self.n_clusters

            self.trace_r = [trace_r, formula_r]

            self.make_dfclusters()
            self.update_appearance_variables()
            self.update_layout_figure()

            if self.widg_colormarkers.value == 'Clustering':
                name_trace = self.name_trace[trace_r]
                with self.fig.batch_update():
                    self.fig.update_traces(
                        selector={'name': name_trace},
                        visible=True
                    )

    def update_point(self, trace, points, selector):
        # changes the points labeled with a cross on the map.

        if not points.point_inds:
            return

        trace = points.trace_index
        formula = self.fig.data[trace].text[points.point_inds[0]]

        if self.widg_checkbox_l.value:
            self.trace_l = [trace, formula]
            self.replica_l = 0
        if self.widg_checkbox_r.value:
            self.trace_r = [trace, formula]
            self.replica_r = 0

        self.make_dfclusters()
        self.update_appearance_variables()
        self.update_layout_figure()

        # if self.widg_checkbox_l.value:
        #     self.widg_compound_text_l.value = formula
        #     self.view_structure_l(formula)
        # if self.widg_checkbox_r.value:
        #     self.widg_compound_text_r.value = formula
        #     self.view_structure_r(formula)

    # def view_structure_l(self, formula):
    #     replicas = self.df.loc[self.df['Formula'] == formula].index.shape[0]
    #     if self.replica_l >= replicas:
    #         self.replica_l = 0
    #     i_structure = self.df.loc[self.df['Formula'] == formula]['File-id'].values[self.replica_l]
    #     self.viewer_l.script("load data/query_nomad_archive/structures/" + str(int(i_structure)) + ".xyz")

    # def view_structure_r(self, formula):
    #     replicas = self.df[self.df['Formula'] == formula].index.shape[0]
    #     if self.replica_r >= replicas:
    #         self.replica_r = 0
    #     i_structure = self.df.loc[self.df['Formula'] == formula]['File-id'].values[self.replica_r]
    #     self.viewer_r.script("load data/query_nomad_archive/structures/" + str(int(i_structure)) + ".xyz")

    def handle_fontfamily_change(self, change):
        self.fig.update_layout(
            font=dict(family=change.new)
        )

    def handle_fontsize_change(self, change):
        self.fig.update_layout(
            font=dict(size=change.new)
        )

    def handle_markersize_change(self, change):
        self.marker_size = int(change.new)
        self.update_appearance_variables()
        self.update_layout_figure()

    def handle_crossize_change(self, change):
        self.cross_size = int(change.new)
        self.update_appearance_variables()
        self.update_layout_figure()

    def handle_colorpalette_change(self, change):
        self.palette = cycle(getattr(px.colors.qualitative, change.new))
        if self.widg_colormarkers.value == 'Clustering':
            with self.fig.batch_update():
                for cl in np.arange(self.n_clusters+1):
                    self.fig.data[cl].update(marker_color=next(self.palette))

    def handle_markercolor_change(self, change):
        if change.new == 'Clustering':
            self.widg_outliersbox.layout.visibility = 'hidden'
            self.widg_continuouscolors.layout.visibility = 'hidden'
            self.widg_addoutliers_label.layout.visibility = 'hidden'
        else:
            self.widg_outliersbox.layout.visibility = 'visible'
            self.widg_continuouscolors.layout.visibility = 'visible'
            self.widg_addoutliers_label.layout.visibility = 'visible'
        self.make_dfclusters()
        self.update_appearance_variables()
        self.update_layout_figure()

    def handle_continuouscolor_change(self, change):
        self.make_dfclusters()
        self.update_appearance_variables()
        self.update_layout_figure()

    def handle_addoutliers_change(self, change):
        self.make_dfclusters()
        self.update_appearance_variables()
        self.update_layout_figure()

    def bgtoggle_button_clicked(self, button):
        if self.bg_toggle:
            self.bg_toggle = False
            self.fig.update_layout(
                plot_bgcolor='white',
                xaxis=dict(gridcolor='rgb(229,236,246)', showgrid=True, zeroline=False),
                yaxis=dict(gridcolor='rgb(229,236,246)', showgrid=True, zeroline=False),
            )
        else:
            self.bg_toggle = True
            self.fig.update_layout(
                plot_bgcolor=self.widg_bgcolor.value,
                xaxis=dict(gridcolor='white'),
                yaxis=dict(gridcolor='white')
            )

    def bgcolor_button_clicked(self, button):
        if self.bg_toggle:
            self.fig.update_layout(plot_bgcolor=self.widg_bgcolor.value)

    def print_button_clicked(self, button):
        self.widg_print_out.clear_output()
        text = "A download link will appear soon."
        with self.widg_print_out:
            print(text)
        path = "./data/query_nomad_archive/plots/"
        try:
            os.mkdir(path)
        except:
            pass
        file_name = self.widg_plot_name.value + '.' + self.widg_plot_format.value
        self.fig.write_image(path + file_name, scale=self.widg_scale.value)
        self.widg_print_out.clear_output()
        with self.widg_print_out:
            local_file = FileLink(path + file_name, result_html_prefix="Click here to download: ")
            display(local_file)

    def plotappearance_button_clicked(self, button):
        if self.widg_box_utils.layout.visibility == 'visible':
            self.widg_box_utils.layout.visibility = 'hidden'
            for i in range(290, -1, -1):
                self.widg_box_viewers.layout.top = str(i) + 'px'

            self.widg_box_utils.layout.bottom = '0px'
        else:
            for i in range(291):
                self.widg_box_viewers.layout.top = str(i) + 'px'
            self.widg_box_utils.layout.bottom = '460px'
            self.widg_box_utils.layout.visibility = 'visible'

    def handle_checkbox_l(self, change):
        if change.new:
            self.widg_checkbox_r.value = False
        else:
            self.widg_checkbox_r.value = True

    def handle_checkbox_r(self, change):
        if change.new:
            self.widg_checkbox_l.value = False
        else:
            self.widg_checkbox_l.value = True

    def handle_frac_change(self, change):
        self.frac = change.new
        self.make_dfclusters()
        self.update_appearance_variables()
        self.update_layout_figure()

    def updatefrac_button_clicked(self, button):
        self.frac = self.widg_frac_slider.value
        self.make_dfclusters()
        self.update_appearance_variables()
        self.update_layout_figure()

    def reset_symbols_button_clicked(self, button):
        self.reset_markers()
        self.update_layout_figure()

    def view(self):

        for name in self.name_trace:
            self.trace[name].on_click(self.update_point)

        self.widg_display_button_l.on_click(self.display_button_l_clicked)
        self.widg_display_button_r.on_click(self.display_button_r_clicked)
        self.widg_checkbox_l.observe(self.handle_checkbox_l, names='value')
        self.widg_checkbox_r.observe(self.handle_checkbox_r, names='value')
        self.widg_print_button.on_click(self.print_button_clicked)
        self.widg_bgtoggle_button.on_click(self.bgtoggle_button_clicked)
        self.widg_bgcolor_button.on_click(self.bgcolor_button_clicked)
        self.widg_markersize.observe(self.handle_markersize_change, names='value')
        self.widg_crosssize.observe(self.handle_crossize_change, names='value')
        self.widg_fontfamily.observe(self.handle_fontfamily_change, names='value')
        self.widg_fontsize.observe(self.handle_fontsize_change, names='value')
        self.widg_plotutils_button.on_click(self.plotappearance_button_clicked)
        self.widg_colorpalette.observe(self.handle_colorpalette_change, names='value')
        self.widg_frac_slider.observe(self.handle_frac_change, names='value')
        self.widg_colormarkers.observe(self.handle_markercolor_change, names='value')
        self.widg_continuouscolors.observe(self.handle_continuouscolor_change, names='value')
        self.widg_outliersbox.observe(self.handle_addoutliers_change, names='value')
        self.widg_reset_button.on_click(self.reset_symbols_button_clicked)
        self.widg_box_utils.layout.visibility = 'hidden'

        self.widg_plotutils_button.layout.left = '50px'
        self.widg_box_utils.layout.border = 'dashed 1px'
        self.widg_box_utils.right = '100px'
        self.widg_box_utils.layout.max_width = '700px'

        # with self.output_l:
        #     display(self.viewer_l)
        # with self.output_r:
        #     display(self.viewer_r)
        container = widgets.VBox([
            widgets.HBox([self.widg_label_colormarkers, self.widg_colormarkers,
                          self.widg_continuouscolors, self.widg_outliersbox,
                          self.widg_addoutliers_label]),
            widgets.HBox([self.widg_label_frac, self.widg_frac_slider, self.widg_update_frac_button]),
            self.fig,
            self.widg_plotutils_button,
            # self.widg_box_viewers,
            self.widg_box_utils
        ])
        display(container)

    def instantiate_widgets(self):

        self.widg_update_frac_button = widgets.Button(
            description='Click to update',
            layout=widgets.Layout(width='150px', left='130px')
        )
        self.widg_frac_slider = widgets.BoundedFloatText(
            min=0,
            max=1,
            step=0.01,
            value=self.frac,
            layout=widgets.Layout(left='130px', width='60px')
        )
        self.widg_label_frac = widgets.Label(
            value='Fraction of compounds visualized in the map: ',
            layout=widgets.Layout(left='130px')
        )
        self.widg_colormarkers = widgets.Dropdown(
            options=self.color_features,
            value=self.color_features[0],
            layout=widgets.Layout(left='130px', width='220px')
        )
        self.widg_continuouscolors = widgets.Dropdown(
            options=px.colors.named_colorscales(),
            value='viridis',
            layout=widgets.Layout(left='130px', width='150px',visibility='hidden')
        )
        self.widg_outliersbox = widgets.Checkbox(
            value=False,
            indent=False,
            layout=widgets.Layout(width='50px', left='130px', visibility='hidden')
        )
        self.widg_addoutliers_label = widgets.Label(
            value='Include outliers',
            layout=widgets.Layout(left='100px', visibility='hidden')
        )
        self.widg_label_colormarkers = widgets.Label(
            value='Marker colors: ',
            layout=widgets.Layout(left='130px', width='100px')
        )
        self.widg_compound_text_l = widgets.Combobox(
            placeholder='...',
            description='Compound:',
            options=self.df_grouped.index.tolist(),
            disabled=False,
            layout=widgets.Layout(width='200px')
        )
        self.widg_compound_text_r = widgets.Combobox(
            placeholder='...',
            description='Compound:',
            options=self.df_grouped.index.tolist(),
            disabled=False,
            layout=widgets.Layout(width='200px')
        )
        self.widg_display_button_l = widgets.Button(
            description="Display",
            layout=widgets.Layout(width='100px')
        )
        self.widg_display_button_r = widgets.Button(
            description="Display",
            layout=widgets.Layout(width='100px')
        )
        self.widg_checkbox_l = widgets.Checkbox(
            value=True,
            indent=False,
            layout=widgets.Layout(width='50px')
        )
        self.widg_checkbox_r = widgets.Checkbox(
            value=False,
            indent=False,
            layout=widgets.Layout(width='50px'),
        )
        self.widg_markersize = widgets.BoundedIntText(
            placeholder=str(self.marker_size),
            description='Marker size',
            value=str(self.marker_size),
            layout=widgets.Layout(left='30px', width='200px')
        )
        self.widg_crosssize = widgets.BoundedIntText(
            placeholder=str(self.cross_size),
            description='Cross size',
            value=str(self.cross_size),
            layout=widgets.Layout(left='30px', width='200px')
        )
        self.widg_fontsize = widgets.BoundedIntText(
            placeholder=str(self.font_size),
            description='Font size',
            value=str(self.font_size),
            layout=widgets.Layout(left='30px', width='200px')
        )
        self.widg_fontfamily = widgets.Dropdown(
            options=self.font_families,
            description='Font family',
            value=self.font_families[0],
            layout=widgets.Layout(left='30px', width='200px')
        )
        self.widg_colorpalette = widgets.Dropdown(
            options=self.qualitative_colors,
            description='Color palette',
            value=self.qualitative_colors[0],
            layout=widgets.Layout(left='30px', width='200px')
        )
        self.widg_bgcolor = widgets.Text(
            placeholder=str(self.bg_color),
            description='Color',
            value=str(self.bg_color),
            layout=widgets.Layout(left='30px', width='200px'),
        )
        self.widg_bgtoggle_button = widgets.Button(
            description='Toggle on/off background',
            layout=widgets.Layout(left='50px', width='200px'),
        )
        self.widg_bgcolor_button = widgets.Button(
            description='Update background color',
            layout=widgets.Layout(left='50px', width='200px'),
        )
        self.widg_reset_button = widgets.Button(
            description='Reset symbols',
            layout=widgets.Layout(left='50px', width='200px')
        )
        self.widg_plot_name = widgets.Text(
            placeholder='plot',
            value='plot',
            description='Name',
            layout=widgets.Layout(width='300px')
        )
        self.widg_plot_format = widgets.Text(
            placeholder='png',
            value='png',
            description='Format',
            layout=widgets.Layout(width='150px')
        )
        self.widg_scale = widgets.Text(
            placeholder='1',
            value='1',
            description="Scale",
            layout=widgets.Layout(width='150px')
        )
        self.widg_print_button = widgets.Button(
            description='Print',
            layout=widgets.Layout(left='50px', width='600px')
        )
        self.widg_print_out = widgets.Output(
            layout=widgets.Layout(left='100px', width='500px')
        )
        self.widg_printdescription = widgets.Label(
            value="Click 'Print' to export the plot in the desired format.",
            layout=widgets.Layout(left='50px', width='640px')
        )
        self.widg_printdescription2 = widgets.Label(
            value="The resolution of the image can be increased by increasing the 'Scale' value.",
            layout=widgets.Layout(left='50px', width='640px')
        )
        self.widg_featuredescription = widgets.Label(
            value="The dropdown menus select the features to visualize."
        )
        self.widg_description = widgets.Label(
            value='Tick the box next to the cross symbols in order to choose which windows visualizes the next '
                  'structure selected in the map above.'
        )
        self.widg_plotutils_button = widgets.Button(
            description='For a high-quality print of the plot, click to access the plot appearance utils',
            layout=widgets.Layout(width='600px')
        )
        self.widg_box_utils = widgets.VBox([widgets.HBox([self.widg_colorpalette,
                                                          self.widg_markersize, self.widg_fontsize,
                                                          ]),
                                            widgets.HBox([self.widg_reset_button, self.widg_crosssize,
                                                          self.widg_fontfamily,
                                                          ]),
                                            widgets.HBox([self.widg_bgtoggle_button, self.widg_bgcolor,
                                                          self.widg_bgcolor_button]),
                                            self.widg_printdescription, self.widg_printdescription2,
                                            widgets.HBox(
                                                [self.widg_plot_name, self.widg_plot_format, self.widg_scale]),
                                            self.widg_print_button, self.widg_print_out,
                                            ])

        file1 = open("./assets/query_nomad_archive/cross.png", "rb")
        image1 = file1.read()
        self.widg_img1 = widgets.Image(
            value=image1,
            format='png',
            width=30,
            height=30,
        )
        file2 = open("./assets/query_nomad_archive/cross2.png", "rb")
        image2 = file2.read()
        self.widg_img2 = widgets.Image(
            value=image2,
            format='png',
            width=30,
            height=30,
        )
        self.output_l = widgets.Output()
        self.output_r = widgets.Output()

        self.widg_box_viewers = widgets.VBox([self.widg_description, widgets.HBox([
            widgets.VBox([
                widgets.HBox([self.widg_compound_text_l, self.widg_display_button_l,
                              self.widg_img1, self.widg_checkbox_l]),
                self.output_l]),
            widgets.VBox(
                [widgets.HBox([self.widg_compound_text_r, self.widg_display_button_r,
                               self.widg_img2, self.widg_checkbox_r]),
                 self.output_r])
        ])])
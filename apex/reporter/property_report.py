import numpy as np
from abc import ABC, abstractmethod
import plotly.graph_objs as go
from dash import dash_table
import pandas as pd

from apex.core.lib.utils import round_format, round_2d_format

TABLE_WIDTH = '50%'
TABLE_MIN_WIDTH = '95%'


def random_color():
    r = np.random.randint(50, 200)
    g = np.random.randint(50, 200)
    b = np.random.randint(50, 200)
    return f'rgb({r}, {g}, {b})'


class PropertyReport(ABC):
    @staticmethod
    @abstractmethod
    def plotly_graph(res_data: dict, name: str):
        """
        Plot plotly graph.

        Parameters
        ----------
        res_data : dict
            The dict storing the result of the props
        Returns:
        -------
        list[plotly.graph_objs]
            The list of plotly graph object
        plotly.graph_objs.layout
            the layout
        """
        pass

    @staticmethod
    @abstractmethod
    def dash_table(res_data: dict, decimal: int) -> [dash_table.DataTable, pd.DataFrame]:
        """
        Make Dash table.

        Parameters
        ----------
        res_data : dict
            The dict storing the result of the props
        Returns:
        -------
        dash_table.DataTable
            The dash table object
        pd.DataFrame
        """
        pass


class EOSReport(PropertyReport):
    @staticmethod
    def plotly_graph(res_data: dict, name: str, **kwargs):
        vpa = []
        epa = []
        for k, v in res_data.items():
            vpa.append(k)
            epa.append(v)
        df = pd.DataFrame({
            "VpA(A^3)": vpa,
            "EpA(eV)": epa
        })
        trace = go.Scatter(
            name=name,
            x=df['VpA(A^3)'],
            y=df['EpA(eV)'],
            mode='lines+markers'
        )
        layout = go.Layout(
            title='Energy of State',
            xaxis=dict(
                title_text="VpA (A<sup>3</sup>)",
                title_font=dict(
                    size=18,
                    color="#7f7f7f"
                )
            ),
            yaxis=dict(
                title_text="EpA (eV)",
                title_font=dict(
                    size=18,
                    color="#7f7f7f"
                )
            )
        )

        return [trace], layout

    @staticmethod
    def dash_table(res_data: dict, decimal: int = 3, **kwargs) -> dash_table.DataTable:
        vpa = []
        epa = []
        for k, v in res_data.items():
            vpa.append(float(k))
            epa.append(float(v))
        df = pd.DataFrame({
            "VpA(A^3)": round_format(vpa, decimal),
            "EpA(eV)": round_format(epa, decimal)
        })

        table = dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'width': TABLE_WIDTH,
                         'minWidth': TABLE_MIN_WIDTH,
                         'overflowX': 'auto'},
            style_cell={'textAlign': 'left'}
        )

        return table, df

class DecohesionEnergyReport(PropertyReport):
    @staticmethod
    def plotly_graph(res_data: dict, name: str, **kwargs):
        decohesion_e = [values[0] for values in res_data.values()]
        stress = [values[1] for values in res_data.values()]
        vacuum_size = [values[2] for values in res_data.values()]
        vacuum_size = [str(item) for item in vacuum_size]
        df = pd.DataFrame({
            "separation distance (A)": vacuum_size,
            "Decohesion energy (J/m^2)": decohesion_e,
            "Decohesion stress (GPa)": [s / 1e9 for s in stress],
        })
        trace_E = go.Scatter(
            name=f"{name} Decohesion Energy",
            x=df['separation distance (A)'],
            y=df['Decohesion energy (J/m^2)'],
            mode='lines+markers',
            yaxis='y1'
        )

        trace_S = go.Scatter(
            name=f"{name} Decohesion Stress",
            x=df['separation distance (A)'],
            y=df['Decohesion stress (GPa)'],
            mode='lines+markers',
            yaxis='y2'
        )
        layout = go.Layout(
            title=dict(
                text='Decohesion Energy and Stress',
                x=0.5,  # 标题居中
                xanchor='center'
            ),
            xaxis=dict(
                title_text="separation distance (A)",
                title_font=dict(
                    size=18,
                    color="#7f7f7f"
                )
            ),
            yaxis=dict(
                title="Decohesion energy (J/m^2)",
                titlefont=dict(
                    size=18,
                    color="#7f7f7f"
                )
            ),
            yaxis2=dict(
                title="Decohesion stress (GPa)",
                titlefont=dict(
                    size=18,
                    color="#7f7f7f"
                ),
                overlaying='y',
                side='right'
            )
        )
        trace = [trace_E, trace_S]
        return trace, layout

    @staticmethod
    def dash_table(res_data: dict, decimal: int = 3, **kwargs) -> dash_table.DataTable:
        decohesion_e = [values[0] for values in res_data.values()]
        stress = [values[1] for values in res_data.values()]
        vacuum_size = [values[2] for values in res_data.values()]
        vacuum_size = [str(item) for item in vacuum_size]
        df = pd.DataFrame({
            "separation distance (A)": vacuum_size,
            "Decohesion energy (J/m^2)": round_format(decohesion_e, decimal),
            "Decohesion stress (GPa)": round_format([s / 1e9 for s in stress], decimal),
        })
        table = dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'width': TABLE_WIDTH,
                         'minWidth': TABLE_MIN_WIDTH,
                         'overflowX': 'auto'},
            style_cell={'textAlign': 'left'}
        )
        return table, df


class Lat_param_T_Report(PropertyReport):
    @staticmethod
    def plotly_graph(res_data: dict, name: str, **kwargs):
        lx = [values[0] for values in res_data.values()]
        ly = [values[1] for values in res_data.values()]
        lz = [values[2] for values in res_data.values()]

        temp = [values[3] for values in res_data.values()]
        temp = [str(item) for item in temp]

        trace_a = go.Scatter(x=temp, y=lx, mode='lines+markers', name='lx', line=dict(color='blue'))
        trace_b = go.Scatter(x=temp, y=ly, mode='lines+markers', name='ly', line=dict(color='green'))
        trace_c = go.Scatter(x=temp, y=lz , mode='lines+markers', name='lz', line=dict(color='red'))

        trace = [trace_a, trace_b, trace_c]

        layout = go.Layout(
            title='Lat_param_T',
            xaxis=dict(title='temperature (K)', tickvals=temp),
            yaxis=dict(title='lattice length (Å)'),
            showlegend=True
        )
        return trace, layout

    @staticmethod
    def dash_table(res_data: dict, decimal: int = 6, **kwargs) -> dash_table.DataTable:
        lx = [values[0] for values in res_data.values()]
        ly = [values[1] for values in res_data.values()]
        lz = [values[2] for values in res_data.values()]
        temp = [values[3] for values in res_data.values()]
        temp = [str(item) for item in temp]
        df = pd.DataFrame({
            "temperature (K)": temp,
            "lx (A)": round_format(lx, decimal),
            "ly (A)": round_format(ly, decimal),
            "lz (A)": round_format(lz, decimal),
        })
        table = dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'width': TABLE_WIDTH,
                         'minWidth': TABLE_MIN_WIDTH,
                         'overflowX': 'auto'},
            style_cell={'textAlign': 'left'}
        )
        return table, df

class ElasticReport(PropertyReport):
    @staticmethod
    def plotly_graph(res_data: dict, name: str, **kwargs):
        elastic_tensor = res_data['elastic_tensor']
        c11 = elastic_tensor[0][0]
        c12 = elastic_tensor[0][1]
        c13 = elastic_tensor[0][2]
        c22 = elastic_tensor[1][1]
        c23 = elastic_tensor[1][2]
        c33 = elastic_tensor[2][2]
        c44 = elastic_tensor[3][3]
        c55 = elastic_tensor[4][4]
        c66 = elastic_tensor[5][5]
        BV = res_data['B']
        GV = res_data['G']
        EV = res_data['E']
        uV = res_data['u']

        polar = go.Scatterpolar(
            name=name,
            r=[c11, c12, c13, c22, c23, c33,
               c44, c55, c66, BV, GV, EV, uV],
            theta=['C11', 'C12', 'C13', 'C22', 'C23', 'C33',
                   'C44', 'C55', 'C66', 'B', 'G', 'E', 'u'],
            fill='none'
        )

        layout = go.Layout(
            showlegend=True,
            title='Elastic Property'
        )

        return [polar], layout

    @staticmethod
    def dash_table(res_data: dict, decimal: int = 3, **kwargs) -> dash_table.DataTable:
        ph = '-'
        et = res_data['elastic_tensor']
        BV = res_data['B']
        GV = res_data['G']
        EV = res_data['E']
        uV = res_data['u']
        null_t = [' '] * 6
        BV_t = ['B', BV, ph, ph, ph, ph]
        GV_t = ['G', GV, ph, ph, ph, ph]
        EV_t = ['E', EV, ph, ph, ph, ph]
        uV_t = ['u', uV, ph, ph, ph, ph]
        table_tensor = [et[0], et[1], et[2], et[3], et[4], et[5],
                        null_t, BV_t, GV_t, EV_t, uV_t]

        # round numbers in table
        rounded_tensor = round_2d_format(table_tensor, decimal)

        df = pd.DataFrame(
            rounded_tensor,
            columns=['Col 1', 'Col 2', 'Col 3', 'Col 4', 'Col 5', 'Col 6'],
        )

        table = dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'width': TABLE_WIDTH,
                         'minWidth': TABLE_MIN_WIDTH,
                         'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'width': '150px'}
        )

        return table, df


class SurfaceReport(PropertyReport):
    @staticmethod
    def plotly_graph(res_data: dict, name: str, **kwargs):
        miller = []
        surf_e = []
        epa = []
        epa_equi = []
        for k, v in res_data.items():
            miller.append(k.split('_')[0])
            surf_e.append(float(v[0]))
            epa.append(float(v[1]))
            epa_equi.append(float(v[2]))

        # enclose polar plot
        surf_e.append(surf_e[0])
        miller.append(miller[0])
        polar = go.Scatterpolar(
            name=name,
            r=surf_e,
            theta=miller,
            fill='none'
        )

        layout = go.Layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    autorange=True
                )
            ),
            showlegend=True,
            title='Surface Forming Energy'
        )

        return [polar], layout

    @staticmethod
    def dash_table(res_data: dict, decimal: int = 3, **kwargs) -> dash_table.DataTable:
        miller = []
        surf_e = []
        epa = []
        epa_equi = []
        for k, v in res_data.items():
            miller.append(k.split('_')[0])
            surf_e.append(float(v[0]))
            epa.append(float(v[1]))
            epa_equi.append(float(v[2]))
        df = pd.DataFrame({
            "Miller Index": miller,
            "E_surface (J/m^2)": round_format(surf_e, decimal),
            "EpA (eV)": round_format(epa, decimal),
            "EpA_equi (eV)": round_format(epa_equi, decimal),
        })

        table = dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'width': TABLE_WIDTH,
                         'minWidth': TABLE_MIN_WIDTH,
                         'overflowX': 'auto'},
            style_cell={'textAlign': 'left'}
        )

        return table, df


class InterstitialReport(PropertyReport):
    @staticmethod
    def plotly_graph(res_data: dict, name: str, **kwargs):
        inter_struct = []
        inter_form_e = []
        struct_e = []
        equi_e = []
        for k, v in res_data.items():
            inter_struct.append(k.split('_')[1])
            inter_form_e.append(float(v[0]))
            struct_e.append(float(v[1]))
            equi_e.append(float(v[2]))

        # enclose polar plot
        inter_struct.append(inter_struct[0])
        inter_form_e.append(inter_form_e[0])

        polar = go.Scatterpolar(
            name=name,
            r=inter_form_e,
            theta=inter_struct,
            fill='none'
        )

        layout = go.Layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    autorange=True
                )
            ),
            showlegend=True,
            title='Interstitial Forming Energy'
        )

        return [polar], layout

    @staticmethod
    def dash_table(res_data: dict, decimal: int = 3, **kwargs) -> dash_table.DataTable:
        inter_struct = []
        inter_form_e = []
        struct_e = []
        equi_e = []
        for k, v in res_data.items():
            inter_struct.append(k.split('_')[1])
            inter_form_e.append(float(v[0]))
            struct_e.append(float(v[1]))
            equi_e.append(float(v[2]))
        df = pd.DataFrame({
            "Initial configuration ": inter_struct,
            "E_form (eV)": round_format(inter_form_e, decimal),
            "E_defect (eV)": round_format(struct_e, decimal),
            "E_equi (eV)": round_format(equi_e, decimal),
        })

        table = dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'width': TABLE_WIDTH,
                         'minWidth': TABLE_MIN_WIDTH,
                         'overflowX': 'auto'},
            style_cell={'textAlign': 'left'}
        )

        return table, df


class VacancyReport(PropertyReport):
    @staticmethod
    def plotly_graph(res_data: dict, name: str, **kwargs):
        v = list(res_data.values())[0]
        vac_form_e = float(v[0])
        struct_e = float(v[1])
        equi_e = float(v[2])

        bar = go.Bar(
            name=name,
            # x=[vac_form_e, struct_e, equi_e],
            # y=['E_form (eV)', 'E_defect (eV)', 'E_equi (eV)'],
            x=[vac_form_e],
            y=['E_form'],
            orientation='h'
        )

        layout = go.Layout(
            title='Vacancy Forming Energy',
            xaxis=dict(
                title_text="Vacancy Forming Energy (eV)",
                title_font=dict(
                    size=18,
                    color="#7f7f7f"
                )
            ),
            yaxis=dict(
                title_text="",
                title_font=dict(
                    size=18,
                    color="#7f7f7f"
                ),
            ),
            showlegend=True
        )

        return [bar], layout

    @staticmethod
    def dash_table(res_data: dict, decimal: int = 3, **kwargs) -> dash_table.DataTable:
        vac_form_e = []
        struct_e = []
        equi_e = []
        for k, v in res_data.items():
            vac_form_e.append(float(v[0]))
            struct_e.append(float(v[1]))
            equi_e.append(float(v[2]))
        df = pd.DataFrame({
            "E_form (eV)": round_format(vac_form_e, decimal),
            "E_defect (eV)": round_format(struct_e, decimal),
            "E_equi (eV)": round_format(equi_e, decimal),
        })

        table = dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'width': TABLE_WIDTH,
                         'minWidth': TABLE_MIN_WIDTH,
                         'overflowX': 'auto'},
            style_cell={'textAlign': 'left'}
        )

        return table, df


class GammaReport(PropertyReport):
    @staticmethod
    def plotly_graph(res_data: dict, name: str, **kwargs):
        displ = []
        displ_length = []
        fault_en = []
        struct_en = []
        equi_en = []
        for k, v in res_data.items():
            displ.append(k)
            displ_length.append(v[0])
            fault_en.append(v[1])
            struct_en.append((v[2]))
            equi_en.append(v[3])
        df = pd.DataFrame({
            "displacement": displ,
            "displace_length": displ_length,
            "fault_en": fault_en
        })
        trace = go.Scatter(
            name=name,
            x=df['displacement'],
            # x=df['displace_length'],
            y=df['fault_en'],
            mode='lines+markers'
        )
        layout = go.Layout(
            title='Stacking Fault Energy (Gamma Line)',
            xaxis=dict(
                title_text="Slip Fraction",
                # title_text="Displace_Length (Å)",
                title_font=dict(
                    size=18,
                    color="#7f7f7f"
                )
            ),
            yaxis=dict(
                title_text='Fault Energy (J/m<sup>2</sup>)',
                title_font=dict(
                    size=18,
                    color="#7f7f7f"
                )
            )
        )

        return [trace], layout

    @staticmethod
    def dash_table(res_data: dict, decimal: int = 3, **kwargs) -> dash_table.DataTable:
        displ = []
        displ_length = []
        fault_en = []
        struct_en = []
        equi_en = []
        for k, v in res_data.items():
            displ.append(float(k))
            displ_length.append(v[0])
            fault_en.append(v[1])
            struct_en.append((v[2]))
            equi_en.append(v[3])
        df = pd.DataFrame({
            "Slip_frac": round_format(displ, decimal),
            "Slip_Length (Å)": round_format(displ_length, decimal),
            "E_Fault (J/m^2)": round_format(fault_en, decimal),
            "E_Slab (eV)": round_format(struct_en, decimal),
            "E_Equilib (eV)": round_format(equi_en, decimal)
        })

        table = dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'width': TABLE_WIDTH,
                         'minWidth': TABLE_MIN_WIDTH,
                         'overflowX': 'auto'},
            style_cell={'textAlign': 'left'}
        )

        return table, df


class PhononReport(PropertyReport):
    @staticmethod
    def plotly_graph(res_data: dict, name: str, **kwargs):
        bands = res_data['band']

        band_path_list = []
        for seg in bands[0]:
            seg_list = [k for k in seg.keys()]
            band_path_list.extend(seg_list)
        band_list = []
        for band in bands:
            seg_result_list = []
            for seg in band:
                seg_result = [v for v in seg.values()]
                seg_result_list.extend(seg_result)
            band_list.append(seg_result_list)
        pd_dict = {"Band Path": band_path_list}
        for ii in range(len(band_list)):
            pd_dict['Band %02d' % (ii + 1)] = band_list[ii]
        df = pd.DataFrame(pd_dict)
        traces = []


        for ii in range(len(band_list)):
            trace = go.Scatter(
                x=df['Band Path'],
                y=df['Band %02d' % (ii + 1)],
                name='Band %02d' % (ii + 1),
                legendgroup=name,
                legendgrouptitle_text=name,
                mode='lines',
                line=dict(color=kwargs["color"], width=1.5)
            )
            traces.append(trace)

        segment_value_list = res_data['segment']
        band_path_info = res_data['band_path']
        segment_value_iter = iter(segment_value_list)

        x_label_list = []
        connect_seg = False
        pre_k = None
        for seg in band_path_info:
            for point in seg:
                k = list(point.keys())[0]
                if connect_seg:
                    new_k = f'{pre_k}/{k}'
                    x_label_list[-1][0] = new_k
                    connect_seg = False
                else:
                    x_label_list.append([k, float(next(segment_value_iter))])
            pre_k = k
            connect_seg = True

        # label special points
        x_label_values_list = [x[1] for x in x_label_list]
        annotations = []
        shapes = []

        for x_label in x_label_list:
            # add label
            annotations.append(go.layout.Annotation(
                x=x_label[1],
                y=1.08,
                xref="x",
                yref="paper",
                text=x_label[0],  # label text
                showarrow=False,
                yshift=0,  # label position
                xanchor='center'
            ))

            # add special vertical line
            '''
            shapes.append({
                'type': 'line',
                'x0': x_label[1],
                'y0': 0,
                'x1': x_label[1],
                'y1': 1,
                'xref': 'x',
                'yref': 'paper',
                'line': {
                    'color': 'grey',
                    'width': 1,
                    'dash': 'dot',
                },
            })
            '''

        layout = go.Layout(
            title='Phonon Spectra',
            annotations=annotations,
            shapes=shapes,
            autotypenumbers='convert types',
            xaxis=dict(
                tickmode='array',
                tickvals=x_label_values_list,
                ticktext=[f'{float(val):.3f}' for val in x_label_values_list],
                title_text="Band Path",
                title_font=dict(
                    size=18,
                    color="#7f7f7f"
                )
            ),
            yaxis=dict(
                title_text="Frequency (THz)",
                title_font=dict(
                    size=18,
                    color="#7f7f7f"
                )
            )
        )

        return traces, layout

    @staticmethod
    def dash_table(res_data: dict, decimal: int = 3, **kwargs) -> dash_table.DataTable:
        bands = res_data['band']
        band_path_list = []
        for seg in bands[0]:
            seg_list = [float(k) for k in seg.keys()]
            band_path_list.extend(seg_list)
            band_path_list.append(' ')
        band_path_list.pop()

        band_list = []
        for band in bands:
            seg_result_list = []
            for seg in band:
                seg_result = [v for v in seg.values()]
                seg_result_list.extend(seg_result)
                seg_result_list.append(' ')
            seg_result_list.pop()
            band_list.append(round_format(seg_result_list, decimal))

        pd_dict = {"Band Path": round_format(band_path_list, decimal)}
        for ii in range(len(band_list)):
            pd_dict['Band %02d' % (ii + 1)] = band_list[ii]

        df = pd.DataFrame(pd_dict)

        table = dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'width': TABLE_WIDTH,
                         'minWidth': TABLE_MIN_WIDTH,
                         'overflowX': 'auto'},
            style_cell={'textAlign': 'left'}
        )

        return table, df


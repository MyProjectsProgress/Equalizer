#  ----------------------------------- JUST  A REFRENCE CODE TO HELP WHILE CREATING SLIDER ---------------------------------------------------------------
# def creating_sliders(names_list):

#     # Side note: we can change sliders colors and can customize sliders as well.
#     # names_list = [('Megzawy', 100),('Magdy', 150)]
#     columns = st.columns(len(names_list))
#     boundary = int(50)
#     sliders_values = []
#     sliders = {}

#     for index, tuple in enumerate(names_list): # ---> [ { 0, ('Megzawy', 100) } , { 1 , ('Magdy', 150) } ]
#         # st.write(index)
#         # st.write(i)
#         min_value = tuple[1] - boundary
#         max_value = tuple[1] + boundary
#         key = f'member{random.randint(0,10000000000)}'
#         with columns[index]:
#             sliders[f'slidergroup{key}'] = svs.vertical_slider(key=key, default_value=tuple[1], step=1, min_value=min_value, max_value=max_value)
#             if sliders[f'slidergroup{key}'] == None:
#                 sliders[f'slidergroup{key}'] = tuple[1]
#             sliders_values.append((tuple[0], sliders[f'slidergroup{key}']))
# names_list = [('A', 100),('B', 150),('C', 75),('D', 25),('E', 150),('F', 60),('G', 86),('H', 150),('E', 150),('G', 25),('K', 99),('L', 150),
#                 ('M', 150),('M', 55),('N', 150)]
# fn.creating_sliders(names_list)

import neurokit2 as nk  # Load the package

simulated_ecg = nk.ecg_simulate(duration=8, sampling_rate=200, heart_rate=80)

nk.signal_plot(simulated_ecg, sampling_rate=200)  # Visualize the signal
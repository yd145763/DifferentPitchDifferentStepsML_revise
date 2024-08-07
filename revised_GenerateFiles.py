# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 18:24:43 2024

@author: limyu
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 12:56:31 2024

@author: limyu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import StrMethodFormatter

from matplotlib.colors import LogNorm

R1 = np.arange(0.6, 1.7, 0.2)
R = pd.Series(R1)
R = R.round(1)


#-------------------------------------contour plots -----------------------------------
for r in R:
    url = "https://raw.githubusercontent.com/yd145763/DifferentPitchML/main/pitch"+str(r)+"um.csv"
    df1 = pd.read_csv(url)
    df1 = df1.iloc[:, 1:]
    x1 = np.linspace(0, 80, len(df1.columns))
    y1 = np.linspace(0, 60, len(df1.index.tolist()))
    
    colorbarmax = df1.max().max()
    
    X,Y = np.meshgrid(x1,y1)
    fig = plt.figure(figsize=(18, 4))
    ax = plt.axes()
    cp=ax.contourf(X,Y,df1, 200, zdir='z', offset=-100, cmap='viridis')
    clb=fig.colorbar(cp, ticks=(np.around(np.linspace(0.0, colorbarmax, num=6), decimals=3)).tolist())
    clb.ax.set_title('Electric Field (eV)', fontweight="bold", fontsize=25)
    for l in clb.ax.yaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_fontsize(30)
    ax.set_xlabel('x-position (µm)', fontsize=30, fontweight="bold", labelpad=1)
    ax.set_ylabel('z-position (µm)', fontsize=30, fontweight="bold", labelpad=1)
    ax.xaxis.label.set_fontsize(30)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_fontsize(30)
    ax.yaxis.label.set_weight("bold")
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.set_yticklabels(ax.get_yticks(), weight='bold')
    ax.set_xticklabels(ax.get_xticks(), weight='bold')
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.show()
    plt.close()
    

#-------------------------------------e field from different pitches -----------------------------------
n = 194
fig = plt.figure(figsize=(20, 13))
ax = plt.axes()
for r in R:
    url = "https://raw.githubusercontent.com/yd145763/DifferentPitchML/main/pitch"+str(r)+"um.csv"
    df1 = pd.read_csv(url)
    df1 = df1.iloc[:, 1:]
    e = df1.iloc[n, :]
    ax.plot(x1, e, linewidth = 5)

#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(50)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(50)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=50)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
plt.xlabel("x-position (µm)")
plt.ylabel("Loss")
plt.legend(["0.6µm", "0.8µm","1.0µm","1.2µm","1.4µm","1.6µm",], prop={'weight': 'bold','size': 40}, loc = "best")
plt.show()
plt.close()



#-------------------------------------prediction actual data -----------------------------------
data = 'actual', 'pred'
pitch = 1.4
steps =5
#n == 129 is equivalent to z = 20um
#n == 194 is equivalent to z = 30um
#n == 269 is equivalent to z = 40um
#n == 291 is equivalent to z = 45um
#n == 323 is equivalent to z = 50um
#n == 387 is equivalent to z = 60um
n = 387
print(n/387*60)
url = 'https://raw.githubusercontent.com/yd145763/DifferentPitchDifferentStepsML_revise/main/df_actual_steps'+str(steps)+'_pitch'+str(pitch)+'.csv'
df1 = pd.read_csv(url)
df1 = df1.iloc[:, 1:]
actual_e = df1['e'+str(n)]
x = df1.iloc[:, -1]

url = 'https://raw.githubusercontent.com/yd145763/DifferentPitchDifferentStepsML_revise/main/df_pred_steps'+str(steps)+'_pitch'+str(pitch)+'.csv'
df1 = pd.read_csv(url)
df1 = df1.iloc[:, 1:]
pred_e = df1['e'+str(n)]

fig = plt.figure(figsize=(20, 13))
ax = plt.axes()
ax.scatter(x,actual_e, s=50, facecolor='blue', edgecolor='blue')
ax.plot(x,pred_e, color = "red", linewidth = 5)
#graph formatting     
ax.tick_params(which='major', width=5.00)
ax.tick_params(which='minor', width=5.00)
ax.xaxis.label.set_fontsize(50)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(50)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=50)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
plt.xlabel("x (µm)")
plt.ylabel("E-field (eV)")
plt.legend(["Original E-field", "Predicted E-field"], prop={'weight': 'bold','size': 45}, loc = "best")
plt.show()
plt.close()


#-------------------------------------training validation data -----------------------------------

steps = 20
#training - validation
url = 'https://raw.githubusercontent.com/yd145763/DifferentPitchDifferentStepsML_revise/main/df_training_step'+str(steps)+'.csv'
df_training = pd.read_csv(url)
df_training = df_training.iloc[:, 1:]

url = 'https://raw.githubusercontent.com/yd145763/DifferentPitchDifferentStepsML_revise/main/df_validation_step'+str(steps)+'.csv'
df_validation = pd.read_csv(url)
df_validation = df_validation.iloc[:, 1:]

N = np.arange(0,388,1)
N = N[::steps]
Z = np.linspace(0,60,388)
Z = Z[::steps]

difference_list = []
absolute_difference_list = []
for n in N:
    training_loss = df_training['loss'+str(n)]
    validation_loss = df_validation['loss'+str(n)]
    difference = training_loss -validation_loss
    difference = np.mean(difference)
    absolute_difference = np.abs(difference)
    print(n, difference)
    difference_list.append(difference)
    absolute_difference_list.append(absolute_difference)



fig = plt.figure(figsize=(20, 13))
ax = plt.axes()
cp = ax.scatter(Z,difference_list, norm=LogNorm(), c= absolute_difference_list, s=200)
clb=fig.colorbar(cp)
clb.ax.set_title('Absolute\nMean Difference', fontweight="bold", fontsize = 50)
for l in clb.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(50)
#ax.plot(Z,difference_list, color = "red", linewidth = 5)
#graph formatting     
ax.tick_params(which='major', width=5.00)
ax.tick_params(which='minor', width=5.00)
ax.xaxis.label.set_fontsize(50)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(50)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=50)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
plt.xlabel("z (µm)")
plt.ylabel("Mean Difference")
plt.show()
plt.close()


epochs = np.arange(1,101,1)
n = 0
training_loss = df_training['loss'+str(n)]
validation_loss = df_validation['loss'+str(n)]

fig = plt.figure(figsize=(20, 13))
ax = plt.axes()
ax.plot(epochs, training_loss, color = "blue", linewidth = 5)
ax.plot(epochs, validation_loss, color = "red", linewidth = 5)
#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(80)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(80)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=80)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Training loss", "Validation Loss"], prop={'weight': 'bold','size': 80}, loc = "best")
plt.show()
plt.close()

#-------------------------------------different steps -----------------------------------
stepsS = 1, 5, 10, 20
pitch = 1.6
markerS = 'o', '^', 'x', '*' 
N = np.arange(0,388, 1)
Z = np.linspace(0,60,388)

fig = plt.figure(figsize=(20, 13))
ax = plt.axes()

for steps, marker in zip(stepsS, markerS):
    url_actual = 'https://raw.githubusercontent.com/yd145763/DifferentPitchDifferentStepsML_revise/main/df_actual_steps'+str(steps)+'_pitch'+str(pitch)+'.csv'
    df1_actual = pd.read_csv(url_actual)
    df1_actual = df1_actual.iloc[:, 1:]
    
    url_pred = 'https://raw.githubusercontent.com/yd145763/DifferentPitchDifferentStepsML_revise/main/df_pred_steps'+str(steps)+'_pitch'+str(pitch)+'.csv'
    df1_pred = pd.read_csv(url_pred)
    df1_pred = df1_pred.iloc[:, 1:]
    
    actual_pred_difference_list = []
    for n in N:
        actual_e = df1_actual['e'+str(n)]
        x = df1_actual.iloc[:, -1]
        pred_e = df1_pred['e'+str(n)]
        difference = np.abs(np.mean(actual_e - pred_e))
        actual_pred_difference_list.append(difference)
    
    
    ax.scatter(Z,actual_pred_difference_list, s=100, marker = marker)
#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(35)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(35)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=35)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
plt.xlabel("z (µm)")
plt.ylabel("Absolue Mean Differences")
plt.yscale('log')
plt.legend(["1", "5", "10", "20"], prop={'weight': 'bold','size': 35}, loc = "best")
plt.show()
plt.close()


stepsS = 1, 5, 10, 20
pitch = 1.4
markerS = 'o', '^', 'x', '*' 
N = np.arange(0,388, 1)
Z = np.linspace(0,60,388)

fig = plt.figure(figsize=(20, 13))
ax = plt.axes()

for steps, marker in zip(stepsS, markerS):
    url_actual = 'https://raw.githubusercontent.com/yd145763/DifferentPitchDifferentStepsML_revise/main/df_actual_steps'+str(steps)+'_pitch'+str(pitch)+'.csv'
    df1_actual = pd.read_csv(url_actual)
    df1_actual = df1_actual.iloc[:, 1:]
    
    url_pred = 'https://raw.githubusercontent.com/yd145763/DifferentPitchDifferentStepsML_revise/main/df_pred_steps'+str(steps)+'_pitch'+str(pitch)+'.csv'
    df1_pred = pd.read_csv(url_pred)
    df1_pred = df1_pred.iloc[:, 1:]
    
    actual_pred_difference_list = []
    for n in N:
        actual_e = df1_actual['e'+str(n)]
        x = df1_actual.iloc[:, -1]
        pred_e = df1_pred['e'+str(n)]
        difference = np.abs(np.mean(actual_e - pred_e))
        actual_pred_difference_list.append(difference)
    
    
    ax.scatter(Z,actual_pred_difference_list, s=100, marker = marker)
#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(35)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(35)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=35)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
plt.xlabel("z (µm)")
plt.ylabel("Absolue Mean Differences")
plt.yscale('log')
plt.legend(["1", "5", "10", "20"], prop={'weight': 'bold','size': 35}, loc = "best")
plt.show()
plt.close()

Pitch = 1.2, 1.4, 1.6
for p in Pitch:
    pitch = p
    
    stepsS = 1, 5, 10, 20
    markerS = 'o', '^', 'x', '*' 
    
    
    N = np.arange(0,388, 1)
    Z = np.linspace(0,60,388)
    
    fig = plt.figure(figsize=(20, 13))
    ax = plt.axes()
    
    for steps, marker in zip(stepsS, markerS):
        url_actual = 'https://raw.githubusercontent.com/yd145763/DifferentPitchDifferentStepsML_revise/main/df_actual_steps'+str(steps)+'_pitch'+str(pitch)+'.csv'
        df1_actual = pd.read_csv(url_actual)
        df1_actual = df1_actual.iloc[:, 1:]
        
        url_pred = 'https://raw.githubusercontent.com/yd145763/DifferentPitchDifferentStepsML_revise/main/df_pred_steps'+str(steps)+'_pitch'+str(pitch)+'.csv'
        df1_pred = pd.read_csv(url_pred)
        df1_pred = df1_pred.iloc[:, 1:]
        
        actual_pred_ape_list = []
        for n in N:
            actual_e = df1_actual['e'+str(n)]
            x = df1_actual.iloc[:, -1]
            pred_e = df1_pred['e'+str(n)]
            difference = np.abs((actual_e - pred_e) / actual_e) *100
            average_percentage_error = np.mean(difference)
            actual_pred_ape_list.append(average_percentage_error)
        
        ax.scatter(Z[129:],actual_pred_ape_list[129:], s=100, marker = marker)
    
    #graph formatting     
    ax.tick_params(which='major', width=2.00)
    ax.tick_params(which='minor', width=2.00)
    ax.xaxis.label.set_fontsize(50)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_fontsize(50)
    ax.yaxis.label.set_weight("bold")
    ax.tick_params(axis='both', which='major', labelsize=50)
    ax.set_yticklabels(ax.get_yticks(), weight='bold')
    ax.set_xticklabels(ax.get_xticks(), weight='bold')
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    plt.xlabel("z (µm)")
    plt.ylabel("APE (%)")
    plt.legend(["1", "5", "10", "20"], prop={'weight': 'bold','size': 50}, loc = "best")
    plt.show()
    plt.close()


#-----------------------------------------------different pitches ---------------

pitches = 0.8, 1.0, 1.2, 1.4, 1.6
markerS = 'o', '^', 'x', '*', 's' 
steps = 1
fig = plt.figure(figsize=(20, 13))
ax = plt.axes()
for pitch, marker in zip(pitches, markerS):
    url_actual = 'https://raw.githubusercontent.com/yd145763/DifferentPitchDifferentStepsML_revise/main/df_actual_steps'+str(steps)+'_pitch'+str(pitch)+'.csv'
    df1_actual = pd.read_csv(url_actual)
    df1_actual = df1_actual.iloc[:, 1:]
    
    url_pred = 'https://raw.githubusercontent.com/yd145763/DifferentPitchDifferentStepsML_revise/main/df_pred_steps'+str(steps)+'_pitch'+str(pitch)+'.csv'
    df1_pred = pd.read_csv(url_pred)
    df1_pred = df1_pred.iloc[:, 1:]
    
    actual_pred_difference_list = []
    for n in N:
        actual_e = df1_actual['e'+str(n)]
        x = df1_actual.iloc[:, -1]
        pred_e = df1_pred['e'+str(n)]
        difference = np.abs(np.mean(actual_e - pred_e))
        actual_pred_difference_list.append(difference)
    
    
    ax.scatter(Z,actual_pred_difference_list, s=100, marker = marker)
#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(35)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(35)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=35)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
plt.xlabel("z (µm)")
plt.ylabel("Absolue Mean Differences")
plt.yscale('log')
plt.legend(["0.8µm", "1.0µm", "1.2µm", "1.4µm", "1.6µm"], prop={'weight': 'bold','size': 35}, loc = "best")
plt.show()
plt.close()


stepsS = 1, 5, 10, 20
pitches = 0.8, 1.0, 1.2, 1.4, 1.6
markerS = 'o', '^', 'x', '*', 's' 

N = np.arange(0,388, 1)
Z = np.linspace(0,60,388)

fig = plt.figure(figsize=(20, 13))
ax = plt.axes()

for pitch, marker in zip(pitches, markerS):
    url_actual = 'https://raw.githubusercontent.com/yd145763/DifferentPitchDifferentStepsML_revise/main/df_actual_steps'+str(steps)+'_pitch'+str(pitch)+'.csv'
    df1_actual = pd.read_csv(url_actual)
    df1_actual = df1_actual.iloc[:, 1:]
    
    url_pred = 'https://raw.githubusercontent.com/yd145763/DifferentPitchDifferentStepsML_revise/main/df_pred_steps'+str(steps)+'_pitch'+str(pitch)+'.csv'
    df1_pred = pd.read_csv(url_pred)
    df1_pred = df1_pred.iloc[:, 1:]
    
    actual_pred_ape_list = []
    for n in N:
        actual_e = df1_actual['e'+str(n)]
        x = df1_actual.iloc[:, -1]
        pred_e = df1_pred['e'+str(n)]
        difference = np.abs((actual_e - pred_e) / actual_e) *100
        average_percentage_error = np.mean(difference)
        actual_pred_ape_list.append(average_percentage_error)
    
    ax.scatter(Z[129:],actual_pred_ape_list[129:], s=100, marker = marker)
print(actual_pred_ape_list.index(max( actual_pred_ape_list)))
#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(50)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(50)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=50)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
plt.xlabel("z (µm)")
plt.ylabel("APE (%)")
plt.legend(["0.8µm", "1.0µm", "1.2µm", "1.4µm", "1.6µm"], prop={'weight': 'bold','size': 50}, loc = "best")
plt.show()
plt.close()

pitches = 0.8, 1.0, 1.2, 1.4, 1.6
stepsS = 1, 5, 10, 20
N = N[194:]
pitch_list = []
steps_list = []
ape_list = []
for pitch in pitches:
    for steps in stepsS:
        pitch_list.append(pitch)
        steps_list.append(steps)
        
        url_actual = 'https://raw.githubusercontent.com/yd145763/DifferentPitchDifferentStepsML_revise/main/df_actual_steps'+str(steps)+'_pitch'+str(pitch)+'.csv'
        df1_actual = pd.read_csv(url_actual)
        df1_actual = df1_actual.iloc[:, 1:]
        
        url_pred = 'https://raw.githubusercontent.com/yd145763/DifferentPitchDifferentStepsML_revise/main/df_pred_steps'+str(steps)+'_pitch'+str(pitch)+'.csv'
        df1_pred = pd.read_csv(url_pred)
        df1_pred = df1_pred.iloc[:, 1:]
        
        actual_pred_ape_list = []
        for n in N:
            actual_e = df1_actual['e'+str(n)]
            x = df1_actual.iloc[:, -1]
            pred_e = df1_pred['e'+str(n)]
            difference = np.abs((actual_e - pred_e) / actual_e) *100
            average_percentage_error = np.mean(difference)
            actual_pred_ape_list.append(average_percentage_error)
        print(pitch)
        print(steps)
        print(np.mean(actual_pred_ape_list))
        ape_list.append(np.mean(actual_pred_ape_list))
        print(' ')

df_results = pd.DataFrame([])
df_results['steps'] = steps_list
df_results['pitch'] = pitch_list
df_results['ape'] = ape_list

import seaborn as sns

mat = df_results.pivot('steps', 'pitch', 'ape')
mat_list = mat.values.tolist()

plt.figure(figsize=(6, 4))
ax = sns.heatmap(mat_list, annot=True, cmap='viridis', fmt=".1f")

mat.columns
ax.set_xticklabels(mat.columns, fontweight="bold", fontsize = 16)
ax.set_yticklabels(mat.index, fontweight="bold", fontsize = 16)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=16, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
cbar.ax.set_title("APE (%)", fontweight="bold", fontsize = 16)
font = {'color': 'black', 'weight': 'bold', 'size': 16}
ax.set_ylabel("Steps", fontdict=font)
ax.set_xlabel("Pitch (µm)", fontdict=font)
#ax.tick_params(axis='both', labelsize=12, weight='bold')
for i, text in enumerate(ax.texts):
    text.set_fontsize(16)
for i, text in enumerate(ax.texts):
    text.set_fontweight('bold')
plt.show()
plt.close()


#------------------------------------------ log full distribution---------------------

steps = 1
pitch = 1.0

url_actual = 'https://raw.githubusercontent.com/yd145763/DifferentPitchDifferentStepsML_revise/main/df_actual_steps'+str(steps)+'_pitch'+str(pitch)+'.csv'
df1_actual = pd.read_csv(url_actual)
df_actual = df1_actual.iloc[:, 1:]

url_pred = 'https://raw.githubusercontent.com/yd145763/DifferentPitchDifferentStepsML_revise/main/df_pred_steps'+str(steps)+'_pitch'+str(pitch)+'.csv'
df1_pred = pd.read_csv(url_pred)
df_pred = df1_pred.iloc[:, 1:]


df_actual.set_index('x', inplace=True)
df_actual = df_actual.transpose()

df_pred.set_index('x', inplace=True)
df_pred = df_pred.transpose()

y1 = np.linspace(0, 60, df_actual.shape[0])
x1 = df_actual.columns.values

from matplotlib.colors import LogNorm

colorbarmax1 = df_actual.max().max()
X,Y = np.meshgrid(x1,y1)
fig = plt.figure(figsize=(18, 4))
ax = plt.axes()
cp=ax.contourf(X,Y,df_actual, 200, zdir='z', offset=-100, cmap='viridis', norm=LogNorm())
clb1=fig.colorbar(cp)
clb1.ax.set_title('Electric Field (eV)', fontweight="bold", fontsize=20)
for l in clb1.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(20)
ax.set_xlabel('x-position (µm)', fontsize=30, fontweight="bold", labelpad=1)
ax.set_ylabel('z-position (µm)', fontsize=30, fontweight="bold", labelpad=1)
ax.xaxis.label.set_fontsize(30)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(30)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=30)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.show()
plt.close()

colorbarmax2 = df_pred.max().max()
X,Y = np.meshgrid(x1,y1)
fig = plt.figure(figsize=(18, 4))
ax = plt.axes()
cp=ax.contourf(X,Y,df_pred, 200, zdir='z', offset=-100, cmap='viridis', norm=LogNorm())
clb2=fig.colorbar(cp)
clb2.ax.set_title('Electric Field (eV)', fontweight="bold", fontsize=20)
for l in clb2.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(20)
ax.set_xlabel('x-position (µm)', fontsize=30, fontweight="bold", labelpad=1)
ax.set_ylabel('z-position (µm)', fontsize=30, fontweight="bold", labelpad=1)
ax.xaxis.label.set_fontsize(30)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(30)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=30)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.show()
plt.close()



#column = 153 to 1128
#rows = 129 onwards

meow = (abs(df_pred - df_actual)/df_actual)*100
meow.mean().mean()

df_pred_shortened = df_pred.iloc[129:, 153:1128]
df_actual_shortened = df_actual.iloc[129:, 153:1128]

meow_shortened = (abs(df_pred_shortened - df_actual_shortened)/df_actual_shortened)*100
meow_shortened.mean().mean()


colorbarmax1 = df_actual.max().max()
X,Y = np.meshgrid(x1,y1)
fig = plt.figure(figsize=(18, 4))
ax = plt.axes()
cp=ax.contourf(X,Y,df_actual, 200, zdir='z', offset=-100, cmap='viridis')
clb1=fig.colorbar(cp, ticks=(np.around(np.linspace(0.0, colorbarmax1, num=6), decimals=3)).tolist())
clb1.ax.set_title('Electric Field (eV)', fontweight="bold")
for l in clb1.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(15)
ax.set_xlabel('x-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
ax.set_ylabel('z-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
ax.xaxis.label.set_fontsize(20)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(20)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.show()
plt.close()

colorbarmax2 = df_pred.max().max()
X,Y = np.meshgrid(x1,y1)
fig = plt.figure(figsize=(18, 4))
ax = plt.axes()
cp=ax.contourf(X,Y,df_pred, 200, zdir='z', offset=-100, cmap='viridis')
clb2=fig.colorbar(cp, ticks=(np.around(np.linspace(0.0, colorbarmax2, num=6), decimals=3)).tolist())
clb2.ax.set_title('Electric Field (eV)', fontweight="bold")
for l in clb2.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(15)
ax.set_xlabel('x-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
ax.set_ylabel('z-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
ax.xaxis.label.set_fontsize(20)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(20)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.show()
plt.close()

#------------------------------------------ time taken---------------------


steps = 20
#training - validation
url = 'https://raw.githubusercontent.com/yd145763/DifferentPitchDifferentStepsML_revise/main/df_result_step'+str(steps)+'.csv'
df_result = pd.read_csv(url)
print(df_result['time_list'].sum())
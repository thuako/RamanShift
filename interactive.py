import glob
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker

import numpy as np
import sys
import pandas as pd
import seaborn as sns

import os

testfileDir = 'testlog(221125)'
testfiles = glob.glob('./' + testfileDir + '/*')
print(testfiles)

smoothRanges = [2, 5]

fulldataPD = pd.DataFrame()
for file in testfiles:
  dataCSV = pd.read_csv(file)
  defaultFreqBW = dataCSV.iloc[1]['Spectrum'] - dataCSV.iloc[0]['Spectrum'] 
  dataPD = pd.DataFrame()
  for colName in dataCSV.iloc[:, 1:-1].columns:
    slicedPD = dataCSV.loc[:, ['Spectrum', colName]]
    slicedPD.rename(columns={colName:'AV'}, inplace=True)
    slicedPD['AV'] = slicedPD['AV']
    slicedPD['iter'] = int(colName)
    slicedPD['freqBW'] = defaultFreqBW
    dataPD = pd.concat([dataPD, slicedPD])

  dataPD['filename'] = file.split('/')[-1].strip('.csv')
  fulldataPD = pd.concat([fulldataPD, dataPD])

GRMSPD = fulldataPD.groupby(['filename', 'Spectrum', 'freqBW']).apply(lambda x: np.sqrt((x['AV'] ** 2).mean()))
GRMSPD = GRMSPD.to_frame(name='GRMS')
GRMSPD.reset_index(inplace=True)

for targetFile in set(GRMSPD['filename']):
  singlePD = GRMSPD.loc[GRMSPD['filename'] == targetFile]
  for smoothRange in smoothRanges:
    smoothed = {'Spectrum':[], 'GRMS': []}
    for startindex in range(0, len(singlePD), smoothRange):
      rangedPD = singlePD.iloc[startindex:startindex+smoothRange]
      # freqMedian1 = rangedPD['Spectrum'].mean()
      # overallGRMS = np.sqrt((rangedPD['GRMS'] ** 2).sum() / smoothRange)
      freqMedian = rangedPD['Spectrum'].mean()
      overallGRMS = rangedPD['GRMS'].sum()
      smoothed['Spectrum'].append(freqMedian)
      smoothed['GRMS'].append(overallGRMS)
    tmpPD = pd.DataFrame(smoothed)
    tmpPD['freqBW'] = defaultFreqBW * smoothRange
    tmpPD['filename'] = targetFile
    GRMSPD = pd.concat([GRMSPD, tmpPD])

GRMSPD['PSD'] = GRMSPD['GRMS'] ** 2 / GRMSPD['freqBW'] **2

testBW = list(set(GRMSPD['freqBW']))[0]
testfilename = list(set(GRMSPD['filename']))[0]

testdata = GRMSPD.loc[(GRMSPD['filename'] == testfilename) & (GRMSPD['freqBW'] == testBW), ['Spectrum', 'PSD']]

# testy = np.random.randint(0, 100, 1000)
# testx = [i for i in range(1000)]
def Show(testData):
  #     list
  testData['IntSpectrum'] = testData['Spectrum'].apply(int)

  # y = testData.PSD.values
  # x = testData.Spectrum.values
  # display(testData)
  # print(x)
  # print(y)
  # len_y = len(testData)
  # _y = [y[-1]]*len_y

  # fig = plt.figure(figsize=(960/72,360/72))
  fig = plt.figure(figsize=(12, 6))
  ax1 = fig.add_subplot(1,1,1)

  # ax1.plot(x, y, color='blue')
  sns.lineplot(x=testData['Spectrum'], y=testData['PSD'], ax=ax1)
  # line_x = ax1.plot(x, _y, color='skyblue')[0]
  line_y = ax1.axvline(testData.iloc[0].loc['Spectrum'], color='skyblue')

  ax1.set_title('aaa')
  text0 = plt.text(testData.iloc[0].loc['Spectrum'],testData.iloc[0].loc['PSD'], str(testData.iloc[0].loc['PSD']),fontsize = 10)

  def scroll(event):
    axtemp=event.inaxes
    x_min, x_max = axtemp.get_xlim()
    fanwei_x = (x_max - x_min) / 10
    if event.button == 'up':
      axtemp.set(xlim=(x_min + fanwei_x, x_max - fanwei_x))
    elif event.button == 'down':
      axtemp.set(xlim=(x_min - fanwei_x, x_max + fanwei_x))
    fig.canvas.draw_idle()
  #
  def motion(event):
    try:
      roundx = (event.xdata // testBW) * testBW
      temp = testData.loc[testData['IntSpectrum'] == int(roundx)].PSD
      line_y.set_xdata(int(roundx))
      text0.set_position((int(roundx), int(temp)))
      text0.set_text(str(temp))

      fig.canvas.draw_idle() #
    except:
      pass

  fig.canvas.mpl_connect('scroll_event', scroll)
  fig.canvas.mpl_connect('motion_notify_event', motion)

  plt.show()
Show(testdata)  

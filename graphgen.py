import glob
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

testfileDir = 'logdir'
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
      freqMedian = rangedPD['Spectrum'].mean()
      overallGRMS = rangedPD['GRMS'].sum()
      smoothed['Spectrum'].append(freqMedian)
      smoothed['GRMS'].append(overallGRMS)
    tmpPD = pd.DataFrame(smoothed)
    tmpPD['freqBW'] = defaultFreqBW * smoothRange
    tmpPD['filename'] = targetFile
    GRMSPD = pd.concat([GRMSPD, tmpPD])

GRMSPD['PSD'] = GRMSPD['GRMS'] ** 2 / GRMSPD['freqBW'] **2


# testy = np.random.randint(0, 100, 1000)
# testx = [i for i in range(1000)]
def Show(testData):
  testData['IntSpectrum'] = testData['Spectrum'].apply(int)
  fig = plt.figure(figsize=(12, 6))
  ax1 = fig.add_subplot(1,1,1)

  sns.lineplot(data=testData, x='Spectrum', y='PSD', ax=ax1, hue='freqBW')
  maxFreqBW = max(set(testData['freqBW']))
  minFreqBW = min(set(testData['freqBW']))
  maxX = max(set(testData['Spectrum']))
  minX = min(set(testData['Spectrum']))
  maxY = max(set(testData['PSD']))
  minY = min(set(testData['PSD']))


  colors = ['red', 'green', 'orange', 'purple']
  listLineX = []
  listLineY = []
  listText = []
  listBW = []
  for bw, color in zip(set(testData['freqBW']), colors):
    listText.append(plt.text(maxX, maxY, str(int(bw)) + ": " ,fontsize = 10))
    listLineX.append(ax1.axhline(maxY, color=color))
    listLineY.append(ax1.axvline(maxX, color=color))
    listBW.append(bw)

  def scroll(event):
    axtemp=event.inaxes
    x_min, x_max = axtemp.get_xlim()
    fanwei_x = (x_max - x_min) / 10
    maxdelta, mindelta = fanwei_x * (x_max - event.xdata)/(x_max - x_min), fanwei_x * (event.xdata - x_min)/(x_max - x_min)
    if event.button == 'up':
      axtemp.set(xlim=(x_min + mindelta, x_max - maxdelta))
    elif event.button == 'down':
      axtemp.set(xlim=(x_min - mindelta, x_max + maxdelta))
    fig.canvas.draw_idle()
  #
  def motion(event):
      yValues = testData.loc[(testData['Spectrum'] < event.xdata) &(testData['Spectrum'] > (event.xdata - maxFreqBW)) ]
      print(f"PLT LOG##### x position(int): {event.xdata}, matched y value: {len(yValues)}")
      print(yValues)
      for i, (lineX, lineY, text, bw) in enumerate(zip(listLineX, listLineY, listText, listBW)):
        data = yValues.loc[yValues['freqBW'] == bw, ['PSD', 'Spectrum']]
        y = data.iloc[0]['PSD']
        x = data.iloc[0]['Spectrum']
        # print(f"PLT LOG####### set y postion: {y}")
        lineX.set_ydata(y)
        lineY.set_xdata(x)
        # textY = minY*2 if i == 0 else maxY/2
        textY = y + (maxY - y) * (i+1)/ 4
        # print(f"PLT LOG####### textY :{textY}, y : {y}")
        text.set_position((x, textY))
        text.set_text("{} BW: ({:.3f}, {:.6f})".format(int(bw), x, y))

      fig.canvas.draw_idle() #

  fig.canvas.mpl_connect('scroll_event', scroll)
  fig.canvas.mpl_connect('motion_notify_event', motion)

  plt.show()

#testfilename = list(set(GRMSPD['filename']))[0]

for testfilename in set(GRMSPD['filename']):
  testdata = GRMSPD.loc[(GRMSPD['filename'] == testfilename) , ['Spectrum', 'PSD', 'freqBW']]
  Show(testdata)

import pandas as pd
import numpy as np
from fastparquet import ParquetFile


extractor_data = pd.read_csv('/run/user/1000/gvfs/smb-share:server=nas01.local,share=rnd/data/date/date_extractions.csv')
extractor_data['imaginary_id'] = extractor_data['croppedImageId_url'].map(lambda x: x.split('/')[-1])

#text = ParquetFile('/run/user/1000/gvfs/smb-share:server=nas01.local,share=rnd/data/parquet_data/text_extractions_temp.parq').to_pandas()
#text.columns = ['imaginary_id', 'Text']

#df = pd.merge(extractor_data, text, on='imaginary_id')


#print('rows in extractor data: ',len(extractor_data))
#print('rows in text Parquet: ',len(text))
#print('rows in merged df: ',len(df))


extractor_data.loc[extractor_data['conclusion'] == 'N\A', 'conclusion'] = np.nan
extractor_data.loc[extractor_data['conclusionConfidence'] == 'N\A', 'conclusionConfidence'] = np.nan
extractor_data.loc[:, 'conclusionConfidence'] = extractor_data['conclusionConfidence'].astype('float')


## sanity check
high_conf = extractor_data.loc[extractor_data['conclusionConfidence'] > 0.9]
print(high_conf.dropna().shape[0] / extractor_data.shape[0])

print(extractor_data.dropna(subset=['conclusion']).shape[0] / extractor_data.shape[0])

extractor_data['conclusionConfidence'].mean()

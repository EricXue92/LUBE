import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd 
import glob 

# fig = plt.figure(figsize=(4, 4))

# To calculate the mean and std
def calculated_outputs(filenames):
	for filename in filenames:
		outputs = []

		df = pd.read_csv(filename)
		No_PCGrad_df = df.iloc[::2]
		No_PCGrad_df.reset_index(drop=True, inplace=True)

		picp_mean = round(No_PCGrad_df['PICP'].mean() , 2)
		picp_std = round(No_PCGrad_df['PICP'].std() , 2)
		NMPIW_mean = round(No_PCGrad_df['NMPIW'].mean() , 2)
		NMPIW_std = round(No_PCGrad_df['NMPIW'].std() , 2)

		outputs.append({'PCIP_Mean':picp_mean, 'PCIP_Std':picp_std,'NMPIW_Mean':NMPIW_mean,'NMPIW_Std':NMPIW_std})


		PCGrad_df = df.iloc[1:len(df):2]
		PCGrad_df.reset_index(drop=True, inplace=True)
		picp_pcg_mean = round(PCGrad_df['PICP'].mean() , 2)
		picp_pcg_std = round(PCGrad_df['PICP'].std() , 2)
		NMPIW_pcg_mean = round(PCGrad_df['NMPIW'].mean() , 2)
		NMPIW_pcg_std = round(PCGrad_df['NMPIW'].std() , 2)

		outputs.append({'PCIP_Mean':picp_pcg_mean, 'PCIP_Std':picp_pcg_std,'NMPIW_Mean':NMPIW_pcg_mean,'NMPIW_Std':NMPIW_pcg_std})

		res = pd.DataFrame(outputs, index = ['No_PCGrad','PCGrad'])
		name= filename.split('.')[0]
		res.to_csv(f'{name}_results.csv')

def plot_box( to_plot = 'PICP'):

	for index, filename in enumerate(filenames):
		df = pd.read_csv(filename)
		No_PCGrad_df = df.iloc[::2]
	
		No_PCGrad_mean = round(No_PCGrad_df[to_plot].mean() * 100 , 2)
		PCGrad_df = df.iloc[1:len(df):2]
		
		PCGrad_mean = round(PCGrad_df[to_plot].mean() * 100, 2)

		No_PCGrad_df.reset_index(drop=True, inplace=True)
		PCGrad_df .reset_index(drop=True, inplace=True)
		df= pd.concat([No_PCGrad_df[to_plot], PCGrad_df[to_plot] ], ignore_index=True, axis = 1)
		df.columns = ['No_PCGrad','PCGrad']
		plt.title(f'Mean:{No_PCGrad_mean} %         Mean:{PCGrad_mean} %')

		ax = sns.boxplot(data = df)
		ax = sns.swarmplot(data = df, color=".25" )
		plt.savefig(f'{to_plot}_{index+1}.png')
		#plt.legend()
		plt.clf()

def main():
	filenames = sorted(glob.glob('*.csv'))
	calculated_outputs(filenames)
	# plot_box()
	# plot_box(to_plot = 'NMPIW')

if __name__ == "__main__":
	main()



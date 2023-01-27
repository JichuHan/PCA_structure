#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")

from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

from sklearn.decomposition import PCA
from sklearn import preprocessing
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from mpl_toolkits.mplot3d import Axes3D

import os


# In[2]:

# In[78]:


class PCA_Jason():
    
    def __init__(self,route):
        data=pd.read_excel(route,index_col=0)
        data.index=pd.to_datetime(data.index)
        self.data=data
        
    def show_data(self):
        return self.data
    
    def norm_shift(self,norm='No',shift='No'):
        if shift=='Yes':
            self.data=(self.data.shift(-1,axis=0)-self.data).dropna()   
        if norm=='Yes':
            for i in self.data.columns:
                self.data[i]=(self.data[i]-np.mean(self.data[i]))/np.std(self.data[i])
        return self.data
    
    def describe(self):
        #(一)三维期限结构
        fig=plt.figure(figsize=(8,8))
        ax1 = Axes3D(fig)

        for i in range(len(self.data.columns)-1,-1,-1):

            D=self.data.index
            x=np.arange(0,len(D)) 
            y=np.ones(len(self.data.index))*i
            z=self.data[self.data.columns[i]]
            if i<len(self.data.columns)/3:
                ax1.plot3D(x,y,z,'dodgerblue') 
            elif i<len(self.data.columns)*2/3:
                ax1.plot3D(x,y,z,'royalblue') 
            else:
                ax1.plot3D(x,y,z,'navy') 
        
        ax1.set_xlabel('Dates from '+str(self.data.index[0])+' to '+str(self.data.index[-1]), fontsize=10)
        ax1.set_ylabel('Terms from '+str(self.data.columns[0])+' to '+str(self.data.columns[-1]), fontsize=10)
        ax1.set_zlabel('Interest Rate',fontsize=10)
        ax1.set_title('Term Structure',fontsize=15)
        
        #（二）均值方差图
        plt.figure(figsize=(10,7))
        mean=self.data.describe().loc['mean']
        std=self.data.describe().loc['std']
        plt.fill_between(mean.index,mean+std,mean-std,facecolor = 'skyblue', alpha = 0.5)
        plt.scatter(mean.index,mean,color='Blue')
        plt.plot(mean.index,mean,color='Black')
        #plt.plot(mean.index,mean+std)
        #plt.plot(mean.index,mean-std)
        plt.title('Mean(+-std)',fontsize=15,pad=20)

        #（三）返回描述性统计
        return self.data.describe().round(3)
    
    def correlation(self):
        plt.figure(figsize=(12,7))
        sns.heatmap(self.data.corr(),annot=True,cmap="Blues")
        plt.title('Correlation Matrix',fontsize=20,pad=20)
        return self.data.corr().round(3),self.data.cov().round(3)
    
    def ts_test(self,lags=5):
        result_p=[]
        result_t=[]
        print(1)
        for i in range(0,len(self.data.columns)):
            dftest = adfuller(self.data.iloc[:,i])
            Lj=acorr_ljungbox(self.data.iloc[:,i], lags)
            result_t.append([dftest[0]]+list(Lj[0]))
            result_p.append([dftest[1]]+list(Lj[1]))
        ts_test_p=pd.DataFrame(result_p).round(3)
        ts_test_p.index=self.data.columns
        ts_test_p.columns=['ADF','Lj-1','Lj-2','Lj-3','Lj-4','Lj-5']
        ts_test_t=pd.DataFrame(result_t).round(3)
        ts_test_t.index=self.data.columns
        ts_test_t.columns=['ADF','Lj-1','Lj-2','Lj-3','Lj-4','Lj-5']
        return ts_test_p,ts_test_t
    
    def histogram(self):
        j=1
        L=np.ceil(np.sqrt(len(self.data.columns)))
        plt.figure(figsize=(25,25))
        for i in self.data.columns:
            plt.subplot(L,L,j)
            plt.hist(self.data[i],color='royalblue')
            plt.title(i,fontsize=10,pad=15)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            j+=1
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.45)
        plt.show()
        
    def Feature_test(self):
        chi_square_value, p_value = calculate_bartlett_sphericity(self.data)
        chi_square_value=round(chi_square_value,3)
        p_value=round(p_value,3)

        kmo_all,kmo_model=calculate_kmo(self.data)
        kmo_model=round(kmo_model,3)

        Test_Result=[['KMO取样适切性量数','巴特利特球形度检验',''],['','近似卡方','显著性'],[kmo_model,chi_square_value, p_value]]
        Test_Result=pd.DataFrame(Test_Result).T
        return Test_Result
        
    def My_PCA(self,n_features=3):
        pca = PCA()
        col=self.data.columns
        pca.fit(self.data)

        data_cov=self.data.cov()
        eigenvalue,featurevector=np.linalg.eig(data_cov)
        self.ev=eigenvalue
        self.fv=featurevector
        self.n=n_features
        self.pca=pca
        
        #(二)计算方差提取比例
        Prim=[]
        Selected=[]
        for i in range(0,len(col)):
            Selected.append(np.dot(pca.components_[:n_features,i]**2,eigenvalue[:n_features]))
            Prim.append(data_cov.iloc[i,i])
            
        Prim_aft=np.array(Prim)/np.array(Prim)
        Selected_aft=np.array(Selected)/np.array(Prim)

        Factor_Variance=pd.DataFrame([Prim,Selected,Prim_aft,Selected_aft]).T.round(3)
        Factor_Variance.index=col
        Factor_Variance.columns=['初始','提取','调整后初始','调整后提取']
        
        #（三）绘制碎石图
        plt.figure(figsize=(10,7))
        plt.scatter(range(1, len(col)+1), eigenvalue,color='Blue')
        plt.plot(range(1, len(col)+1), eigenvalue,color='Black')
        plt.plot(range(1, len(col)+1),np.ones(len(col)),color='#1E90FF')

        plt.title("Scree Plot",size=15,pad=15)  
        plt.xlabel("Factors",size=12)
        plt.ylabel("Eigenvalue",size=12)

        #plt.grid()  # 显示网格
        plt.show()  # 显示图形
        
        #(四)方差解释
        Explained_Variance_Ratio=pca.explained_variance_ratio_ 
        Cumsum_Ratio=pca.explained_variance_ratio_ .cumsum()
        Total=pca.explained_variance_

        Total_Selected=pca.explained_variance_[:n_features]
        Cumsum_Ratio_Selected=pca.explained_variance_ratio_ .cumsum()[:n_features]
        Explained_Variance_Ratio_Selected=pca.explained_variance_ratio_[:n_features]

        Variance_Explaination=pd.DataFrame([Total,Explained_Variance_Ratio,Cumsum_Ratio,Total_Selected,Explained_Variance_Ratio_Selected,Cumsum_Ratio_Selected]).T

        Variance_Explaination.index=col
        Variance_Explaination.columns=['总计','方差百分比','累积%','总计','方差百分比','累积%']
        Variance_Explaination=Variance_Explaination.round(3)

        Variance_Explaination=Variance_Explaination.fillna('')
        Variance_Explaination
        
        #(五)因子载荷
        def Factor_Loading(featurevector,eigenvalue,data_cov,col):
            corr_prim=[]
            corr_reo=[]
            for i in range(0,len(featurevector)):
                temp=featurevector[i]*np.sqrt(eigenvalue[i])
                temp1=[]
                for j in range(0,len(temp)):
                    temp1.append(temp[j]/np.sqrt(data_cov.iloc[j,j]))
                corr_prim.append(temp)
                corr_reo.append(temp1)

            corr_prim_df=pd.DataFrame(corr_prim)
            corr_prim_df.columns=col
            corr_reo_df=pd.DataFrame(corr_reo)
            corr_reo_df.columns=col
            
            return corr_prim_df,corr_reo_df
        
        #prim是初步的，reo是真正的载荷
        corr_prim,corr_reo=Factor_Loading(pca.components_,eigenvalue,data_cov,col)
        corr_prim=corr_prim.round(3).iloc[:n_features,:]
        corr_reo=corr_reo.round(3).iloc[:n_features,:]
        pca_names=[]
        for i in range(0,len(corr_prim)):
            pca_names.append('PC '+str(i+1))
        corr_prim.index=pca_names
        corr_reo.index=pca_names
        
        #（六）因子载荷热力图
        plt.figure(figsize = (15,5))
        ax = sns.heatmap(corr_reo, annot=True, cmap="Blues")

        ax.yaxis.set_tick_params(labelsize=10)
        plt.title("Factor Loading", fontsize="xx-large",pad=15)

        plt.ylabel("Principal Components", fontsize="x-large")
        plt.show()
        
        #（七）因子载荷时间序列
        corr_reo.iloc[:n_features,:].T.plot(figsize=(15,7))
        plt.title('Factor Loading on Principal Components',size=15,pad=15)
        plt.ylabel('Factor Loading',size=12)
        plt.xlabel('Features',size=12)
        plt.tick_params(axis='x', pad=10)
        #plt.grid()
        
        return Factor_Variance,Variance_Explaination,corr_prim,corr_reo

    def Components(self,diff=['SHIBOR1W','SHIBOR2W'],pc_name='PC1'):
       
        if len(diff)==1:
            D=self.data[diff[0]]
            data_name=diff[0]
        elif len(diff)==2:
            D=self.data[diff[0]]-self.data[diff[1]]
            data_name=diff[0]+'-'+diff[1]
        

        PC=pd.DataFrame(np.dot(self.data,self.pca.components_[:self.n].T))
        PC.columns=['PC'+str(i+1) for i in range(len(PC.columns))]
        PC.index=self.data.index

        plt.figure(figsize=(10,6))
        plt.plot(self.data.index,list(PC[pc_name]),color='royalblue')
        plt.plot(self.data.index,list(D),color='orange')
        plt.xlabel('Date',fontsize=10)
        plt.ylabel('Interest Rate',fontsize=10)
        plt.title('Principal Component and Data Comparison',fontsize=15,pad=15)
        plt.legend([pc_name,data_name])
        plt.show()

        plt.figure(figsize=(10,6))
        plt.scatter(PC[pc_name],D,color='royalblue')
        plt.xlabel(pc_name)
        plt.ylabel(data_name)   
        
        return PC
    
    def LA_test(self):
        data_cov=self.data.cov()
        eigenvalue,featurevector=np.linalg.eig(data_cov)
        return np.dot(data_cov,featurevector[:,1]),eigenvalue[1]*featurevector[:,1],sum(featurevector[:,2]**2)
    
    def PCA_Easy(self,n_features=3):
        Test_result=self.Feature_test()
        Factor_Variance,Variance_Explaination,corr_prim,corr_reo=self.My_PCA(n_features)
        return Test_result,Factor_Variance,Variance_Explaination,corr_prim,corr_reo
    
    def Description_Easy(self):
        description=self.describe()
        corr,cov=self.correlation()
        ts_p,ts_t=self.ts_test(lags=5)
        self.histogram()
        return description,corr,cov,ts_p,ts_t
        


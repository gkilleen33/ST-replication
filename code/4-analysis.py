# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 17:26:35 2021

@author: Grady
"""

import os 
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
from stargazer.stargazer import Stargazer
from scipy.stats.mstats import winsorize
from randomization_inference import ri_p_values, cs_ri_p_values
import multiprocess as mp

#%% 
# Set the working directory to the parent folder of this script 
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir('..')

#%% 
# Import 2014-2018 national level VIIRS nightlights data
viirs_national = pd.read_csv('data/viirs-national.csv')
viirs_national['year'] = pd.to_numeric(viirs_national['Date'].str[0:4])
viirs_national = viirs_national[['country_na', 'year', 'sum']]
viirs_national.rename(columns={'country_na': 'country', 'sum': 'viirs'}, inplace=True)

#%%
# Read in and merge GDP and population data from 2014-2018 
gdp = pd.read_csv('data/national-gdp-2014-2018.csv')
pop = gdp[gdp['Series Name'] == 'Population, total']
gdp = gdp[gdp['Series Name'] == 'GDP per capita (constant 2010 US$)']

gdp.drop(columns=['Country Code', 'Series Name', 'Series Code'], inplace=True)
pop.drop(columns=['Country Code', 'Series Name', 'Series Code'], inplace=True)

pop.rename(columns = {'2014 [YR2014]': 'population_2014', '2015 [YR2015]': 'population_2015',
                           '2016 [YR2016]': 'population_2016', '2017 [YR2017]': 'population_2017',
                           '2018 [YR2018]': 'population_2018'}, inplace=True)


gdp.rename(columns = {'2014 [YR2014]': 'GDP_capita_2014', '2015 [YR2015]': 'GDP_capita_2015',
                           '2016 [YR2016]': 'GDP_capita_2016', '2017 [YR2017]': 'GDP_capita_2017',
                           '2018 [YR2018]': 'GDP_capita_2018'}, inplace=True)

pop.loc[(pop['Country Name'] == 'Egypt, Arab Rep.'), 'Country Name'] = 'Egypt'
gdp.loc[(gdp['Country Name'] == 'Egypt, Arab Rep.'), 'Country Name'] = 'Egypt'

pop.loc[(pop['Country Name'] == 'Eswatini'), 'Country Name'] = 'Swaziland'
gdp.loc[(gdp['Country Name'] == 'Eswatini'), 'Country Name'] = 'Swaziland'

pop.loc[(pop['Country Name'] == 'Congo, Rep.'), 'Country Name'] = 'Rep of the Congo'
gdp.loc[(gdp['Country Name'] == 'Congo, Rep.'), 'Country Name'] = 'Rep of the Congo'

pop.loc[(pop['Country Name'] == 'Congo, Dem. Rep.'), 'Country Name'] = 'Dem Rep of the Congo'
gdp.loc[(gdp['Country Name'] == 'Congo, Dem. Rep.'), 'Country Name'] = 'Dem Rep of the Congo'

pop.loc[(pop['Country Name'] == 'Central African Republic'), 'Country Name'] = 'Central African Rep'
gdp.loc[(gdp['Country Name'] == 'Central African Republic'), 'Country Name'] = 'Central African Rep'

# Reshape to long
gdp = pd.wide_to_long(gdp, stubnames='GDP_capita_', i='Country Name', j='year')
gdp.rename(columns={'GDP_capita_': 'GDP_capita'}, inplace=True)
pop = pd.wide_to_long(pop, stubnames='population_', i='Country Name', j='year')
pop.rename(columns={'population_': 'population'}, inplace=True)

# Merge with VIIRS data
national_data = viirs_national.merge(pop, how='left', left_on=['country', 'year'], right_index=True)
national_data = national_data.merge(gdp, how='left', left_on=['country', 'year'], right_index=True)

# Generate log variables
national_data['GDP_capita'] = pd.to_numeric(national_data['GDP_capita'], errors='coerce')
national_data['population'] = pd.to_numeric(national_data['population'], errors='coerce')
national_data['log_GDP'] = np.log(national_data['GDP_capita']*national_data['population'])
national_data['log_GDP_capita'] = np.log(national_data['GDP_capita'])
national_data['nightlights_capita'] = 10000*national_data['viirs']/national_data['population']

national_data['log_nightlights'] = np.log(1 + national_data['viirs'])  # Smallest value is 0, scaling to avoid log of negatives
national_data['log_nightlights_capita'] = np.log(1 + national_data['nightlights_capita'])  # Smallest value is 0, scaling to avoid log of negatives

national_data['country_id'] = national_data['country'].astype('category')

# Get rid of missing data
national_data.dropna(inplace=True)

viirs_model = smf.ols('log_GDP ~ log_nightlights*C(year)', data=national_data).fit(cov_type='cluster', cov_kwds={'groups': national_data['country_id']})
print(viirs_model.summary())

viirs_capita_model = smf.ols('log_GDP_capita ~ log_nightlights_capita*C(year)', data=national_data).fit(cov_type='cluster', cov_kwds={'groups': national_data['country_id']})
print(viirs_capita_model.summary())

# Export results to latex
def assign_stars(coef, pval):
    if pval < .01:
        return '{:,.3f}'.format(coef) + '\sym{***}'
    elif pval < .05:
        return '{:,.3f}'.format(coef) + '\sym{**}'
    elif pval < 0.1:
        return '{:,.3f}'.format(coef) + '\sym{*}'
    else:
        return '{:,.3f}'.format(coef)
    
to_export = pd.DataFrame(columns=['Var', '(1)', '(2)'])
to_export.loc[0] = 'Constant'
to_export.loc[1] = ''
to_export.loc[2] = ''
to_export.loc[3] = 'Night lights'
to_export.loc[4] = ''
to_export.loc[5] = ''
for y in [2015, 2016, 2017, 2018]:
    to_export.loc[len(to_export)] = str(y)
    to_export.loc[len(to_export)] = ''
    to_export.loc[len(to_export)] = ''
    to_export.loc[len(to_export)] = 'NL x {}'.format(y)
    to_export.loc[len(to_export)] = ''
    to_export.loc[len(to_export)] = ''
to_export.loc[len(to_export)] = 'Observations'
to_export.loc[len(to_export)] = 'Adj $R^2$'
to_export.loc[len(to_export)] = 'Entities'
to_export.loc[len(to_export)] = 'Time periods'

i = 1
for r in [viirs_model, viirs_capita_model]:
    to_export.loc[(to_export['Var'] == 'Observations'), '({})'.format(i)] = '{:,.0f}'.format(r.nobs)
    to_export.loc[(to_export['Var'] == 'Adj $R^2$'), '({})'.format(i)] = '{:,.3f}'.format(r.rsquared_adj)
    to_export.loc[(to_export['Var'] == 'Entities'), '({})'.format(i)] = '{:,.0f}'.format(len(set(national_data['country'])))
    to_export.loc[(to_export['Var'] == 'Time periods'), '({})'.format(i)] = '{:,.0f}'.format(len(set(national_data['year'])))
    to_export.loc[(to_export['Var'] == 'Constant'), '({})'.format(i)] = assign_stars(r.params['Intercept'], r.pvalues['Intercept'])
    to_export.loc[1, '({})'.format(i)] = '({:,.3f})'.format(r.bse['Intercept'])
    to_export.loc[3, '({})'.format(i)] = assign_stars(list(r.params)[5], list(r.pvalues)[5])  # NL
    to_export.loc[4, '({})'.format(i)] = '({:,.3f})'.format(list(r.bse)[5])  # NL SE
    for j in [1, 2, 3, 4]:  # Year and interaction coefficients and SEs
        to_export.loc[6*j, '({})'.format(i)] = assign_stars(list(r.params)[j], list(r.pvalues)[j])
        to_export.loc[6*j + 1, '({})'.format(i)] = '({:,.3f})'.format(list(r.bse)[j])
        to_export.loc[6*j + 3, '({})'.format(i)] = assign_stars(list(r.params)[j + 5], list(r.pvalues)[j + 5])
        to_export.loc[6*j + 4, '({})'.format(i)] = '({:,.3f})'.format(list(r.bse)[j + 5])
    i += 1
with open('tables/viirs.tex', 'w') as f:
    to_export.to_latex(f, header=False, index=False, na_rep='', escape=False)

# Plot log(GDP) against log(NL)
national_data['Year'] = national_data['year']

plt.style.use('seaborn-colorblind')
sns.scatterplot(x='log_nightlights', y='log_GDP', data=national_data, hue='Year')
plt.ylabel('log(GDP)')
plt.xlabel('log(Night lights)')
plt.savefig('figures/log_GDP.png', bbox_inches = 'tight', dpi=200)
plt.show()

sns.scatterplot(x='log_nightlights_capita', y='log_GDP_capita', data=national_data, hue='Year')
plt.ylabel('log(GDP/capita)')
plt.xlabel('log(Night lights/capita)')
plt.savefig('figures/log_GDP_capita.png', bbox_inches = 'tight', dpi=200)
plt.show()

#%% 
# Now read in the nighlights and population data by historical ethnic group, and predict log(GDP/capita)

df = pd.read_csv('data/viirs.csv')
df['year'] = pd.to_numeric(df['Date'].str[:4])
df.drop(columns=['Date'], inplace=True)

zs = pd.read_csv('data/zonal-stats.csv')
zs = pd.wide_to_long(zs, stubnames='population_', i='old_ethnic', j='year')
zs.rename(columns={'population_': 'population'}, inplace=True)

df = df.merge(zs, how='left', left_on=['old_ethnic', 'year'], right_index=True)

# Add GDP/capita predictions 
df['log_nightlights'] = np.log(1 + df['viirs'])
df['nightlights_capita'] = 10000*df['viirs']/df['population']
df['log_nightlights_capita'] = np.log(1 + df['nightlights_capita'])
df['log_gdp'] = viirs_model.predict(df[['log_nightlights', 'year']])
df['log_gdp_capita'] = viirs_capita_model.predict(df[['log_nightlights_capita', 'year']])

# Add rainfall 
temp = pd.read_csv('data/average-rainfall.csv')
temp = temp.groupby('old_ethnic').mean()
df = df.merge(temp, how='left', left_on='old_ethnic', right_index=True)

# Drop observations from areas that were historically uninhabited 
df = df[~df.old_ethnic.str.contains('UNINHABITED')]

# Convert country to a categorical variable 
df['country'] = df['country'].astype('category')
df.rename(columns = {'malaria-pf': 'malaria_pf', 'malaria-pv': 'malaria_pv', 'slave_trad': 'slave_trade'}, inplace=True)

# Set the entity and time index 
df['ethnicity_id'] = df['old_ethnic'].astype('category').cat.codes
df['eth_id'] = df['old_ethnic'].astype('category').cat.codes
df['yr'] = df['year']
df.set_index(['eth_id', 'yr'], inplace=True, drop=True)

#%% 
# Export a cross tab of slave trade vs persistence 
def label_indicator(val):
    if val == 1:
        return 'Yes'
    elif val == 0:
        return 'No'
    else:
        return None
df['Participated in slave trade'] = df.apply(lambda row: label_indicator(row['slave_trade']), axis=1)
df['Persistent ethnic group'] = df.apply(lambda row: label_indicator(row['persistent']), axis=1)

cross_tab = pd.crosstab(df[df['year'] == 2014]['Participated in slave trade'],
                        df[df['year'] == 2014]['Persistent ethnic group'])

cross_tab['Total'] = cross_tab['No'] + cross_tab['Yes']
cross_tab.loc['Total'] = cross_tab.loc['No'] + cross_tab.loc['Yes']

with open('tables/cross_tab.tex', 'w') as f:
    cross_tab.to_latex(f)
    
df['rainfall'] = 24*df['rainfall']  # From mm/hr to mm/day
df['total_ensl'] = (1/1000)*df['total_ensl']  # For display
    
#%% 
# Add summary stats 
def generate_sum_stats(out_path, variables, new_var_names = dict()):
    sum_stats = pd.DataFrame(columns=['Variable', 'Full sample', 'ST = 0', 'ST: 1-0', 'Persistent = 0', 'Persistent: 1-0'])
    for var in variables:
        index_mean = len(sum_stats)
        index_sd = len(sum_stats) + 1
        index_blank = len(sum_stats) + 2  # Blank line for display
        if var in new_var_names.keys():
            sum_stats.loc[index_mean, 'Variable'] = new_var_names[var]
        else:
            sum_stats.loc[index_mean, 'Variable'] = var
        sum_stats.loc[index_mean, 'Full sample'] = '{:,.3f}'.format(df[var].mean())
        sum_stats.loc[index_sd, 'Full sample'] = '[{:,.3f}]'.format(df[var].std())
        
        sum_stats.loc[index_mean, 'ST = 0'] = '{:,.3f}'.format(df[df['slave_trade'] == 0][var].mean())
        sum_stats.loc[index_sd, 'ST = 0'] = '[{:,.3f}]'.format(df[df['slave_trade'] == 0][var].std())
        st_reg = smf.ols('{} ~ slave_trade'.format(var), data=df[df['year'] == 2014]).fit(cov_type='HC0')
        sum_stats.loc[index_mean, 'ST: 1-0'] = assign_stars(st_reg.params['slave_trade'], st_reg.pvalues['slave_trade'])
        sum_stats.loc[index_sd, 'ST: 1-0'] = '({:,.3f})'.format(st_reg.bse['slave_trade'])
        
        sum_stats.loc[index_mean, 'Persistent = 0'] = '{:,.3f}'.format(df[df['persistent'] == 0][var].mean())
        sum_stats.loc[index_sd, 'Persistent = 0'] = '[{:,.3f}]'.format(df[df['persistent'] == 0][var].std())
        st_reg = smf.ols('{} ~ persistent'.format(var), data=df[df['year'] == 2014]).fit(cov_type='HC0')
        sum_stats.loc[index_mean, 'Persistent: 1-0'] = assign_stars(st_reg.params['persistent'], st_reg.pvalues['persistent'])
        sum_stats.loc[index_sd, 'Persistent: 1-0'] = '({:,.3f})'.format(st_reg.bse['persistent'])
        
        sum_stats.loc[index_blank, 'Variable'] = ''
        
    obs_index = len(sum_stats)
    sum_stats.loc[obs_index, 'Variable'] = 'Observations'
    sum_stats.loc[obs_index, 'Full sample'] = '{:,.0f}'.format(len(df[df['year'] == 2014]))
    sum_stats.loc[obs_index, 'ST = 0'] = '{:,.0f}'.format(len(df[(df['year'] == 2014) & (df['slave_trade'] == 0)]))
    sum_stats.loc[obs_index, 'ST: 1-0'] = '{:,.0f}'.format(len(df[df['year'] == 2014]))
    sum_stats.loc[obs_index, 'Persistent = 0'] = '{:,.0f}'.format(len(df[(df['year'] == 2014) & (df['persistent'] == 0)]))
    sum_stats.loc[obs_index, 'Persistent: 1-0'] = '{:,.0f}'.format(len(df[df['year'] == 2014])) 
    
    dep_vars = ' + '.join(variables)
    f_stat_index = len(sum_stats)
    sum_stats.loc[f_stat_index, 'Variable'] = 'p-val joint orthogonality'
    st_reg = smf.ols('slave_trade ~ {}'.format(dep_vars), data=df[df['year'] == 2014]).fit(cov_type='HC0')
    sum_stats.loc[f_stat_index, 'ST: 1-0'] = '{:,.3f}'.format(st_reg.f_pvalue)
    st_reg = smf.ols('persistent ~ {}'.format(dep_vars), data=df[df['year'] == 2014]).fit(cov_type='HC0')
    sum_stats.loc[f_stat_index, 'Persistent: 1-0'] = '{:,.3f}'.format(st_reg.f_pvalue)
    
    if len(new_var_names) > 0:
        for n in new_var_names.keys():
            sum_stats.loc[(sum_stats['Variable'] == n), 'Variable'] = new_var_names[n]
    with open(out_path, 'w') as f:
        sum_stats.to_latex(f, header=False, index=False, na_rep='', escape=False)
        
sum_vars = ['diamonds', 'oil', 'gold_dep', 
            'malaria_pf', 'malaria_pv', 'rainfall', 'neighbors',
            'num_coloni', 'population', 'Belgium', 'Britain', 'France',
            'Germany', 'Italy', 'Portugal', 'Spain']
sum_to_rename = {'diamonds': 'Diamond deposits', 
                'oil': 'Oil', 'gold_dep': 'Gold deposits', 
                'malaria_pf': 'Malaria (Pf)', 'malaria_pv': 'Malaria (Pv)', 
                'rainfall': 'Annual rainfall (mm/day)', 'neighbors': 'Neighbors',
                'num_coloni': 'Colonizers (number)', 'Belgium': 'Belgium (colonized by)',
                'population': 'Population (2014, mil)'}

df['population'] = (1/10**6)*df['population']

generate_sum_stats('tables/summary.tex', sum_vars, sum_to_rename)
    
#%% 

# log(GDP/capita)
ols1 = PanelOLS.from_formula('log_gdp_capita ~ slave_trade + TimeEffects', data=df).fit(cov_type="clustered", cluster_entity=True)
print(ols1)

# log(nightlights)
ols2 = PanelOLS.from_formula('log_gdp ~ slave_trade + TimeEffects', data=df).fit(cov_type="clustered", cluster_entity=True)
print(ols2)

# Add controls 
controls = 'diamonds + oil + gold_dep + malaria_pf + malaria_pv + rainfall + neighbors'
ols3 = PanelOLS.from_formula('log_gdp_capita ~ slave_trade + {} + TimeEffects'.format(controls), data=df).fit(cov_type="clustered", cluster_entity=True)
print(ols3)

ols4 = PanelOLS.from_formula('log_gdp ~ slave_trade + {} + TimeEffects'.format(controls), data=df).fit(cov_type="clustered", cluster_entity=True)
print(ols4)

#%% Define a function to export the PanelOLS results to latex, and do so 

def panel_to_tex(out_path, results, new_var_names=dict(), ri=False, ri_p_vals = None, cross_section=False, rf_bootstrap = None):
    to_export = pd.DataFrame(columns=['Var'])
    var_names = list()
    for r in results:
        for v in r.params.keys():
            if v not in var_names:
                var_names.append(v)
    for v in var_names:
        to_export.loc[len(to_export)] = v
        to_export.loc[len(to_export)] = ''  # For SE
        to_export.loc[len(to_export)] = ''  # Blank line between variables for display
        if ri:
            if ('persistent' in v) or ('slave_trade' in v) or ('continuous_persistence' in v):
                to_export.loc[len(to_export)] = ''  # For RI p-value
    
    fixed_effects = list()
    if not cross_section:
        for r in results:
            for f in r.included_effects:
                if '{} FE'.format(f) not in fixed_effects:
                    fixed_effects.append('{} FE'.format(f))
                    
        for f in fixed_effects:
            to_export.loc[len(to_export)] = f
    
    to_export.loc[len(to_export)] = 'Observations'
    
    if cross_section:
        to_export.loc[len(to_export)] = 'Adj $R^2$'
    else:
        to_export.loc[len(to_export)] = '$R^2$ within'
        to_export.loc[len(to_export)] = 'Entities'
        to_export.loc[len(to_export)] = 'Time periods'
    # We have now populated the labels for each variable. Now iterate over the models and populate the results
    i = 1 
    for r in results:
        to_export['({})'.format(i)] = ''  # Column to store results
        for p in r.params.keys():
            p_index = to_export[to_export['Var'] == p].index[0]
            if rf_bootstrap:
                column = rf_bootstrap[i-1]
                to_export.loc[p_index, '({})'.format(i)] = assign_stars(r.params[p], column[p])
                to_export.loc[p_index + 1, '({})'.format(i)] = '[{:,.3f}]'.format(column[p])
            else:
                to_export.loc[p_index, '({})'.format(i)] = assign_stars(r.params[p],r.pvalues[p])
                if cross_section:
                    to_export.loc[p_index + 1, '({})'.format(i)] = '({:,.3f})'.format(r.bse[p])
                else:
                    to_export.loc[p_index + 1, '({})'.format(i)] = '({:,.3f})'.format(r.std_errors[p])
                if ri:
                    if ('persistent' in p) or ('slave_trade' in p) or ('continuous_persistence' in p):
                        to_export.loc[p_index + 2, '({})'.format(i)] = '[{:,.3f}]'.format(ri_p_vals[i-1][p])
        if not cross_section:
            for f in r.included_effects:
                to_export.loc[(to_export['Var'] == '{} FE'.format(f)), '({})'.format(i)] = 'Yes'
        to_export.loc[(to_export['Var'] == 'Observations'), '({})'.format(i)] = '{:,.0f}'.format(r.nobs)
        
        if cross_section:
            to_export.loc[(to_export['Var'] == 'Adj $R^2$'), '({})'.format(i)] = '{:,.3f}'.format(r.rsquared_adj)
        else:     
            to_export.loc[(to_export['Var'] == '$R^2$ within'), '({})'.format(i)] = '{:,.3f}'.format(r._r2)
            to_export.loc[(to_export['Var'] == 'Entities'), '({})'.format(i)] = '{:,.0f}'.format(r.entity_info['total'])
            to_export.loc[(to_export['Var'] == 'Time periods'), '({})'.format(i)] = '{:,.0f}'.format(r.time_info['total'])
        i += 1
    if len(new_var_names) > 0:
        for n in new_var_names.keys():
            to_export.loc[(to_export['Var'] == n), 'Var'] = new_var_names[n]
    with open(out_path, 'w') as f:
        to_export.to_latex(f, header=False, index=False, na_rep='', escape=False)
        
to_rename = {'slave_trade': 'Slave trade', 'diamonds': 'Diamond deposits', 'oil': 'Oil', 'gold_dep': 'Gold deposits', 
             'malaria_pf': 'Malaria (Pf)', 'malaria_pv': 'Malaria (Pv)', 'rainfall': 'Annual rainfall (mm/day)', 
             'neighbors': 'Neighbors'}
panel_to_tex('tables/ols.tex', [ols1, ols2, ols3, ols4], to_rename)

#%% 
# Now estimate a diff in diff model 
# log(GDP/capita)
dd_ri_p = list()

dd1 = PanelOLS.from_formula('log_gdp_capita ~ slave_trade * persistent + TimeEffects', data=df).fit(cov_type="clustered", cluster_entity=True)
dd_ri_p.append(ri_p_values('log_gdp_capita ~ slave_trade * persistent', df, dd1.params, 10000))
print(dd1)

# log(nightlights)
dd2 = PanelOLS.from_formula('log_gdp ~ slave_trade * persistent + TimeEffects', data=df).fit(cov_type="clustered", cluster_entity=True)
dd_ri_p.append(ri_p_values('log_gdp ~ slave_trade * persistent', df, dd2.params, 10000))
print(dd2)

# Add controls 
dd3 = PanelOLS.from_formula('log_gdp_capita ~ slave_trade * persistent + {} + TimeEffects'.format(controls), data=df).fit(cov_type="clustered", cluster_entity=True)
dd_ri_p.append(ri_p_values('log_gdp_capita ~ slave_trade * persistent + {}'.format(controls), df, dd3.params, 10000))
print(dd3)

dd4 = PanelOLS.from_formula('log_gdp ~ slave_trade * persistent + {} + TimeEffects'.format(controls), data=df).fit(cov_type="clustered", cluster_entity=True)
dd_ri_p.append(ri_p_values('log_gdp ~ slave_trade * persistent + {}'.format(controls), df, dd4.params, 10000))
print(dd4)

# Now restrict to West Africa only 
west_africa = df.loc[df.west_afric == 1, :]  # a cutoff by character limit in zonal stats
west_africa.set_index(['ethnicity_id', 'year'], inplace=True, drop=False)

dd5 = PanelOLS.from_formula('log_gdp_capita ~ slave_trade * persistent + {} + TimeEffects'.format(controls), data=west_africa).fit(cov_type="clustered", cluster_entity=True)
dd_ri_p.append(ri_p_values('log_gdp_capita ~ slave_trade * persistent + {}'.format(controls), west_africa, dd5.params, 10000))
print(dd5)

dd6 = PanelOLS.from_formula('log_gdp ~ slave_trade * persistent + {} + TimeEffects'.format(controls), data=west_africa).fit(cov_type="clustered", cluster_entity=True)
dd_ri_p.append(ri_p_values('log_gdp ~ slave_trade * persistent + {}'.format(controls), west_africa, dd6.params, 10000))
print(dd6)

to_rename['persistent'] = 'Persistent ethnicity'
to_rename['slave_trade:persistent'] = 'Slave trade x Persistent'
panel_to_tex('tables/diff-in-diff.tex', [dd1, dd2, dd3, dd4, dd5, dd6], to_rename, ri=True, ri_p_vals=dd_ri_p)

#%% 
# Appendix: Diff-in-diff using only 2018 data
# log(GDP/capita)
df_cs = df.groupby('ethnicity_id').agg('mean')
dd_cs_ri_p = list()

dd_cs_1 = smf.ols('log_gdp_capita ~ slave_trade * persistent', data=df_cs).fit(cov_type='HC0')
dd_cs_ri_p.append(cs_ri_p_values('log_gdp_capita ~ slave_trade * persistent', df_cs, dd_cs_1.params, 10000))
print(dd_cs_1.summary())

# log(nightlights)
dd_cs_2 = smf.ols('log_gdp ~ slave_trade * persistent', data=df_cs).fit(cov_type='HC0')
dd_cs_ri_p.append(cs_ri_p_values('log_gdp ~ slave_trade * persistent', df_cs, dd_cs_2.params, 10000))
print(dd_cs_2.summary())

# Add controls 
dd_cs_3 = smf.ols('log_gdp_capita ~ slave_trade * persistent + {}'.format(controls), data=df_cs).fit(cov_type='HC0')
dd_cs_ri_p.append(cs_ri_p_values('log_gdp_capita ~ slave_trade * persistent + {}'.format(controls), df_cs, dd_cs_3.params, 10000))
print(dd_cs_3.summary())

dd_cs_4 = smf.ols('log_gdp ~ slave_trade * persistent + {}'.format(controls), data=df_cs).fit(cov_type='HC0')
dd_cs_ri_p.append(cs_ri_p_values('log_gdp ~ slave_trade * persistent + {}'.format(controls), df_cs, dd_cs_4.params, 10000))
print(dd_cs_4.summary())

# Now restrict to West Africa only 
west_africa_cs = df_cs.loc[df_cs.west_afric == 1, :]
dd_cs_5 = smf.ols('log_gdp_capita ~ slave_trade * persistent + {}'.format(controls), data=west_africa_cs).fit(cov_type='HC0')
dd_cs_ri_p.append(cs_ri_p_values('log_gdp_capita ~ slave_trade * persistent + {}'.format(controls), west_africa_cs, dd_cs_5.params, 10000))
print(dd5)

dd_cs_6 = smf.ols('log_gdp ~ slave_trade * persistent + {}'.format(controls), data=west_africa_cs).fit(cov_type='HC0')
dd_cs_ri_p.append(cs_ri_p_values('log_gdp ~ slave_trade * persistent + {}'.format(controls), west_africa_cs, dd_cs_6.params, 10000))
print(dd6)

to_rename['persistent'] = 'Persistent ethnicity'
to_rename['slave_trade:persistent'] = 'Slave trade x Persistent'
panel_to_tex('tables/diff-in-diff-cs.tex', [dd_cs_1, dd_cs_2, dd_cs_3, dd_cs_4, dd_cs_5, dd_cs_6], to_rename, ri=True, ri_p_vals=dd_cs_ri_p, cross_section=True)

#%% 
# Examine the ability to predict persistence of ethnic groups (note: only use 1 year of data since no nightlights)
df['slaves_area'] = df['total_ensl'] / df['area']

pers_controls = controls + ' + num_coloni + Belgium + Britain + France + Germany + Italy + Portugal + Spain'

pers1 = smf.ols('persistent ~ slave_trade', data=df[df['year'] == 2014]).fit(cov_type='HC0')
print(pers1.summary())

pers2 = smf.ols('persistent ~ slave_trade + {}'.format(pers_controls), data=df[df['year'] == 2014]).fit(cov_type='HC0')
print(pers2.summary())

pers3 = smf.ols('persistent ~ slaves_area', data=df[df['year'] == 2014]).fit(cov_type='HC0')
print(pers3.summary())

pers4 = smf.ols('persistent ~ slaves_area + {}'.format(pers_controls), data=df[df['year'] == 2014]).fit(cov_type='HC0')
print(pers4.summary())

pers5 = smf.ols('persistent ~ slaves_area', data=df[(df['year'] == 2014) & (df['slave_trade'] == 1)]).fit(cov_type='HC0')
print(pers5.summary())

pers6 = smf.ols('persistent ~ slaves_area + {}'.format(pers_controls), df[(df['year'] == 2014) & (df['slave_trade'] == 1)]).fit(cov_type='HC0')
print(pers6.summary())

results = Stargazer([pers1, pers2, pers3, pers4, pers5, pers6])
results.covariate_order(['slave_trade', 'slaves_area', 'diamonds', 'oil', 'gold_dep', 
                         'malaria_pf', 'malaria_pv', 'rainfall', 'neighbors',
                         'num_coloni', 'Belgium', 'Britain', 'France',
                         'Germany', 'Italy', 'Portugal', 'Spain'])
results.rename_covariates({'slaves_area': 'Slaves/area', 'slave_trade': 'Slave trade',
                           'diamonds': 'Diamond deposits', 
                           'oil': 'Oil', 'gold_dep': 'Gold deposits', 
                           'malaria_pf': 'Malaria (Pf)', 'malaria_pv': 'Malaria (Pv)', 
                           'rainfall': 'Annual rainfall (mm/day)', 'neighbors': 'Neighbors',
                           'num_coloni': 'Colonizers (number)', 'Belgium': 'Belgium (colonized by)'})

with open('tables/persistence.tex', 'w') as f:
    f.write(results.render_latex())

#%% 
# For appendix: regression on raw nightlights 
nl_ri_p = list()

# log(NL/capita)
nl1 = PanelOLS.from_formula('log_nightlights_capita ~ slave_trade * persistent + TimeEffects', data=df).fit(cov_type="clustered", cluster_entity=True)
nl_ri_p.append(ri_p_values('log_nightlights_capita ~ slave_trade * persistent', df, nl1.params, 10000))
print(nl1)

# log(nightlights)
nl2 = PanelOLS.from_formula('log_nightlights ~ slave_trade * persistent + TimeEffects', data=df).fit(cov_type="clustered", cluster_entity=True)
nl_ri_p.append(ri_p_values('log_nightlights ~ slave_trade * persistent', df, nl2.params, 10000))
print(nl2)

# Add controls 
nl3 = PanelOLS.from_formula('log_nightlights_capita ~ slave_trade * persistent + {} + TimeEffects'.format(controls), data=df).fit(cov_type="clustered", cluster_entity=True)
nl_ri_p.append(ri_p_values('log_nightlights_capita ~ slave_trade * persistent + {}'.format(controls), df, nl3.params, 10000))
print(nl3)

nl4 = PanelOLS.from_formula('log_nightlights ~ slave_trade * persistent + {} + TimeEffects'.format(controls), data=df).fit(cov_type="clustered", cluster_entity=True)
nl_ri_p.append(ri_p_values('log_nightlights ~ slave_trade * persistent + {}'.format(controls), df, nl4.params, 10000))
print(nl4)

# Now restrict to West Africa only 
nl5 = PanelOLS.from_formula('log_nightlights_capita ~ slave_trade * persistent + {} + TimeEffects'.format(controls), data=west_africa).fit(cov_type="clustered", cluster_entity=True)
nl_ri_p.append(ri_p_values('log_nightlights_capita ~ slave_trade * persistent + {}'.format(controls), west_africa, nl5.params, 10000))
print(nl3)

nl6 = PanelOLS.from_formula('log_nightlights ~ slave_trade * persistent + {} + TimeEffects'.format(controls), data=west_africa).fit(cov_type="clustered", cluster_entity=True)
nl_ri_p.append(ri_p_values('log_nightlights ~ slave_trade * persistent + {}'.format(controls), west_africa, nl6.params, 10000))
print(nl4)

to_rename['persistent'] = 'Persistent ethnicity'
to_rename['slave_trade:persistent'] = 'Slave trade x Persistent'
panel_to_tex('tables/diff-in-diff-nl.tex', [nl1, nl2, nl3, nl4, nl5, nl6], to_rename, ri=True, ri_p_vals=nl_ri_p)

#%% 

from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


# Now attempt to predict persistence with a random forest among groups that did not participate in the slave trade
df_2014 = df[(df['year'] == 2014) & (df['slave_trade'] == 0)]

# First use all observations for hyperparameter searching using CV
X = df_2014[['diamonds', 'oil', 'gold_dep', 'malaria_pf', 'malaria_pv', 'rainfall', 'surface',
           'num_coloni', 'lat', 'lon', 'Belgium', 'Britain', 'France', 'Germany', 'Italy', 'Portugal', 'Spain']]
features = list(X.columns)
X = X.values

y = df_2014[['persistent']].values.ravel()

def evaluate_model(predictions, probs, y_test):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    baseline['recall'] = recall_score(y_test, [1 for _ in range(len(y_test))])
    baseline['precision'] = precision_score(y_test, [1 for _ in range(len(y_test))])
    baseline['roc'] = 0.5
        
    results = {}
    
    results['recall'] = recall_score(y_test, predictions)
    results['precision'] = precision_score(y_test, predictions)
    results['roc'] = roc_auc_score(y_test, probs)
    
    print(results)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
    model_fpr, model_tpr, _ = roc_curve(y_test, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, '--', color = '#FFC20A', label = 'baseline')
    plt.plot(model_fpr, model_tpr, color = '#0C7BDC', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves');

# Hyperparameter grid
param_grid = {
    'n_estimators': np.linspace(10, 100).astype(int),
    'max_depth': [None] + list(np.linspace(3, 20).astype(int)),
    'max_features': ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1)),
    'max_leaf_nodes': [None] + list(np.linspace(10, 50).astype(int)),
    'min_samples_split': [2, 3, 4, 5],
    'class_weight': ['balanced', 'balanced_subsample'],
    'ccp_alpha': np.arange(0.0, 0.2, 0.01)
}

# Estimator for use in random search
estimator = RandomForestClassifier(random_state = 0)

# Create the random search model
rs = RandomizedSearchCV(estimator, param_grid, n_jobs = -1, 
                        scoring = 'roc_auc', cv = 10, 
                        n_iter = 100, verbose = 1, random_state=0)

rs.fit(X, y)

# Print score of best estimator
print(rs.best_score_)

params = rs.best_params_
print(params)

rf = RandomForestClassifier(random_state = 0, n_jobs=-1, **params)

# Add groups to dataframe for k-fold cross validation (5 approximately equal sized groups partitioned by slave trade, equal assignment for a given ethnicity_id)
df_rf = df.copy(deep = True)
df_rf['fold'] = None
df_rf.sort_values(by=['slave_trade', 'ethnicity_id', 'year'], inplace=True)
df_rf['row_by_st'] = df_rf.groupby(['slave_trade', 'year']).cumcount() + 1
df_rf.loc[df_rf['slave_trade'] == 0, 'fold'] = pd.cut(df_rf[df_rf['slave_trade'] == 0]['row_by_st'], bins=5, labels=False)
df_rf.loc[df_rf['slave_trade'] == 1, 'fold'] = pd.cut(df_rf[df_rf['slave_trade'] == 1]['row_by_st'], bins=5, labels=False)

# Now randomly reshuffle the folds within slave trade status, keeping groups fixed
import random
ids_1 = df_rf.loc[df_rf['slave_trade'] == 0, 'ethnicity_id'].unique()
random.Random(0).shuffle(ids_1)

ids_2 = df_rf.loc[df_rf['slave_trade'] == 1, 'ethnicity_id'].unique()
random.Random(0).shuffle(ids_2)

ids = list(ids_1) + list(ids_2)  # So shuffling occurs by group and within slave trade status 

temp = df_rf.copy(deep=True)
temp = temp.set_index('ethnicity_id').loc[ids].reset_index()

df_rf['fold'] = list(temp['fold'])

results = df_rf.copy(deep=True)
results = results.loc[:, ['ethnicity_id', 'year', 'fold', 'persistent', 'slave_trade']]
results['predicted_persistence'] = None
results['predicted_probabilities'] = None

# Now train, predict, and score using 5-fold CV 
for k in range(5):
    df_k = df_rf.loc[df_rf['fold'] == k]
    df_nk = df_rf.loc[df_rf['fold'] != k]
    
    # Extract the 2014 observations with slave_trade = 0 in groups nk for training 
    X_nk = df_nk[(df_nk['year'] == 2014) & (df_nk['slave_trade'] == 0)]
    X_nk = X_nk[['diamonds', 'oil', 'gold_dep', 'malaria_pf', 'malaria_pv', 'rainfall', 'surface',
                 'num_coloni', 'lat', 'lon', 'Belgium', 'Britain', 'France', 'Germany', 
                 'Italy', 'Portugal', 'Spain']]
    y_nk = df_nk[(df_nk['year'] == 2014) & (df_nk['slave_trade'] == 0)]
    y_nk = y_nk[['persistent']].values.ravel()

    # Train model 
    rf.fit(X_nk, y_nk)
    
    # Now predict on fold k 
    X_k = df_k[['diamonds', 'oil', 'gold_dep', 'malaria_pf', 'malaria_pv', 'rainfall', 'surface',
                'num_coloni', 'lat', 'lon', 'Belgium', 'Britain', 'France', 
                'Germany', 'Italy', 'Portugal', 'Spain']].values
    results.loc[results['fold'] == k, 'predicted_persistence'] = rf.predict(X_k)
    results.loc[results['fold'] == k, 'predicted_probabilities'] = rf.predict_proba(X_k)[:, 1]

results['predicted_persistence'] = results['predicted_persistence'].astype('int64')

evaluate_model(results.loc[(results['year'] == 2014) & (results['slave_trade'] == 0), 'predicted_persistence'], 
               results.loc[(results['year'] == 2014) & (results['slave_trade'] == 0), 'predicted_probabilities'], 
               results.loc[(results['year'] == 2014) & (results['slave_trade'] == 0), 'persistent'])

plt.savefig('figures/roc.png', bbox_inches = 'tight', dpi=200)
plt.show()

#%% Now look at a diff in diff with predicted persistence 

df = df.merge(results[['ethnicity_id', 'year', 'predicted_persistence']], on=['ethnicity_id', 'year'])

# Set the entity and time index 
df['eth_id'] = df['ethnicity_id']
df['yr'] = df['year']
df.set_index(['eth_id', 'yr'], inplace=True, drop=True)

dd_pred1 = PanelOLS.from_formula('log_gdp_capita ~ slave_trade * predicted_persistence + TimeEffects', data=df).fit(cov_type="clustered", cluster_entity=True)
print(dd_pred1)

dd_pred2 = PanelOLS.from_formula('log_gdp ~ slave_trade * predicted_persistence + TimeEffects', data=df).fit(cov_type="clustered", cluster_entity=True)
print(dd_pred2)

# Add controls 
dd_pred3 = PanelOLS.from_formula('log_gdp_capita ~ slave_trade * predicted_persistence + {} + TimeEffects'.format(controls), data=df).fit(cov_type="clustered", cluster_entity=True)
print(dd_pred3)

dd_pred4 = PanelOLS.from_formula('log_gdp ~ slave_trade * predicted_persistence + {} + TimeEffects'.format(controls), data=df).fit(cov_type="clustered", cluster_entity=True)
print(dd_pred4)

# Calculate bootstrapped p-values
df_reindex = df.set_index(['ethnicity_id', 'year'], drop=False, verify_integrity=True)
df_reindex.drop(columns=['predicted_persistence'], inplace=True)
ethnicities_ids = pd.DataFrame(df.ethnicity_id.unique())

dd_pred1_bootstraps = pd.DataFrame(columns = list(dd_pred1.params.keys()))
dd_pred2_bootstraps = pd.DataFrame(columns = list(dd_pred2.params.keys()))
dd_pred3_bootstraps = pd.DataFrame(columns = list(dd_pred3.params.keys()))
dd_pred4_bootstraps = pd.DataFrame(columns = list(dd_pred4.params.keys()))

for i in range(1, 10001):
    rf.random_state = i  # Change the random state in the Random Forest model to account for Monte Carlo uncertainty
    
    # Draw a bootstrap sample (sample with replacemenet from ID, but keep all years for each ID draw)
    bootstrap = ethnicities_ids.sample(n=ethnicities_ids.shape[0], replace=True, random_state=i)
    bootstrap.rename(columns={bootstrap.columns[0]: 'ethnicity_id'}, inplace=True)
    years = list(range(2014, 2019))
    list_of_ids = bootstrap.ethnicity_id.to_list()
    multiindex = [list_of_ids, np.array(years)]
    bootstrap_sample_index = pd.MultiIndex.from_product(multiindex, names=['ethnicity_id', 'year'])
    bootstrap_sample = df_reindex.loc[bootstrap_sample_index, :]
    bootstrap_sample.reset_index(inplace=True, drop=True)
    
    # Estimate the model, using 5-fold CV again
    
    # Add groups to dataframe for k-fold cross validation (5 approximately equal sized groups partitioned by slave trade, equal assignment for a given ethnicity_id)
    bootstrap_sample['fold'] = None
    bootstrap_sample.sort_values(by=['slave_trade', 'ethnicity_id', 'year'], inplace=True)
    bootstrap_sample['row_by_st'] = bootstrap_sample.groupby(['slave_trade', 'year']).cumcount() + 1
    bootstrap_sample.loc[bootstrap_sample['slave_trade'] == 0, 'fold'] = pd.cut(bootstrap_sample[bootstrap_sample['slave_trade'] == 0]['row_by_st'], bins=5, labels=False)
    bootstrap_sample.loc[bootstrap_sample['slave_trade'] == 1, 'fold'] = pd.cut(bootstrap_sample[bootstrap_sample['slave_trade'] == 1]['row_by_st'], bins=5, labels=False)
    
    # Now randomly reshuffle the folds within slave trade status, keeping groups fixed
    ids_1 = bootstrap_sample.loc[bootstrap_sample['slave_trade'] == 0, 'ethnicity_id'].unique()
    random.Random(0).shuffle(ids_1)
    
    ids_2 = bootstrap_sample.loc[bootstrap_sample['slave_trade'] == 1, 'ethnicity_id'].unique()
    random.Random(0).shuffle(ids_2)
    
    ids = list(ids_1) + list(ids_2)  # So shuffling occurs by group and within slave trade status 
    
    temp = bootstrap_sample.copy(deep=True)
    temp = temp.set_index('ethnicity_id').loc[ids].reset_index()
    
    bootstrap_sample['fold'] = list(temp['fold'])
    
    results = bootstrap_sample.copy(deep=True)
    results = results.loc[:, ['ethnicity_id', 'year', 'fold', 'persistent', 'slave_trade']]
    results['predicted_persistence'] = None
    results['predicted_probabilities'] = None
    
    # Now train, predict, and score using 5-fold CV 
    for k in range(5):
        df_k = bootstrap_sample.loc[bootstrap_sample['fold'] == k]
        df_nk = bootstrap_sample.loc[bootstrap_sample['fold'] != k]
        
        # Extract the 2014 observations with slave_trade = 0 in groups nk for training 
        X_nk = df_nk[(df_nk['year'] == 2014) & (df_nk['slave_trade'] == 0)]
        X_nk = X_nk[['diamonds', 'oil', 'gold_dep', 'malaria_pf', 'malaria_pv', 'rainfall', 'surface',
                     'num_coloni', 'lat', 'lon', 'Belgium', 'Britain', 'France', 'Germany', 
                     'Italy', 'Portugal', 'Spain']]
        y_nk = df_nk[(df_nk['year'] == 2014) & (df_nk['slave_trade'] == 0)]
        y_nk = y_nk[['persistent']].values.ravel()
    
        # Train model 
        rf.fit(X_nk, y_nk)
        
        # Now predict on fold k 
        X_k = df_k[['diamonds', 'oil', 'gold_dep', 'malaria_pf', 'malaria_pv',  'rainfall', 'surface',
                    'num_coloni', 'lat', 'lon', 'Belgium', 'Britain', 'France', 
                    'Germany', 'Italy', 'Portugal', 'Spain']].values
        results.loc[results['fold'] == k, 'predicted_persistence'] = rf.predict(X_k)
        results.loc[results['fold'] == k, 'predicted_probabilities'] = rf.predict_proba(X_k)[:, 1]
    
    results['predicted_persistence'] = results['predicted_persistence'].astype('int64')
    
    # Add predicted persistence to the bootstrapped sample 
    bootstrap_sample = bootstrap_sample.merge(results[['ethnicity_id', 'year', 'predicted_persistence']], on=['ethnicity_id', 'year'])
    
    bootstrap_sample.set_index(['ethnicity_id', 'year'], inplace=True, drop=False)
    
    # Estimate each of the 4 regressions 
    _beta = PanelOLS.from_formula('log_gdp_capita ~ slave_trade * predicted_persistence + TimeEffects', data=bootstrap_sample).fit().params
    dd_pred1_bootstraps.loc[i] =  list(_beta)
    
    _beta = PanelOLS.from_formula('log_gdp ~ slave_trade * predicted_persistence + TimeEffects', data=bootstrap_sample).fit().params
    dd_pred2_bootstraps.loc[i] =  list(_beta)
    
    _beta = PanelOLS.from_formula('log_gdp_capita ~ slave_trade * predicted_persistence + {} + TimeEffects'.format(controls), data=bootstrap_sample).fit().params
    dd_pred3_bootstraps.loc[i] =  list(_beta)
    
    _beta = PanelOLS.from_formula('log_gdp ~ slave_trade * predicted_persistence + {} + TimeEffects'.format(controls), data=bootstrap_sample).fit().params
    dd_pred4_bootstraps.loc[i] =  list(_beta)
    
# Now calculate the p-values for each model 
dd_pred1_p = dict()
dd_pred2_p = dict()
dd_pred3_p = dict()
dd_pred4_p = dict()
    
j = 0
for var in list(dd_pred1.params.keys()):
    num_larger = dd_pred1_bootstraps[abs(dd_pred1_bootstraps[var] - dd_pred1_bootstraps[var].mean()) > abs(dd_pred1.params[j])][var].count()
    p = num_larger/len(dd_pred1_bootstraps)
    dd_pred1_p[var] = p
    j += 1  
    
j = 0
for var in list(dd_pred2.params.keys()):
    num_larger = dd_pred2_bootstraps[abs(dd_pred2_bootstraps[var] - dd_pred2_bootstraps[var].mean()) > abs(dd_pred2.params[j])][var].count()
    p = num_larger/len(dd_pred2_bootstraps)
    dd_pred2_p[var] = p
    j += 1  
    
j = 0
for var in list(dd_pred3.params.keys()):
    num_larger = dd_pred3_bootstraps[abs(dd_pred3_bootstraps[var] - dd_pred3_bootstraps[var].mean()) > abs(dd_pred3.params[j])][var].count()
    p = num_larger/len(dd_pred3_bootstraps)
    dd_pred3_p[var] = p
    j += 1  
    
j = 0
for var in list(dd_pred4.params.keys()):
    num_larger = dd_pred4_bootstraps[abs(dd_pred4_bootstraps[var] - dd_pred4_bootstraps[var].mean()) > abs(dd_pred4.params[j])][var].count()
    p = num_larger/len(dd_pred4_bootstraps)
    dd_pred4_p[var] = p
    j += 1  

to_rename['predicted_persistence'] = 'Predicted persistence'
to_rename['slave_trade:predicted_persistence'] = 'Slave trade x Predicted persistence'
panel_to_tex('tables/diff-in-diff-pred.tex', [dd_pred1, dd_pred2, dd_pred3, dd_pred4], to_rename,
             rf_bootstrap=[dd_pred1_p, dd_pred2_p, dd_pred3_p, dd_pred4_p])

#%% 
# Winsorized diff in diff 
df['log_gdp_capita_wins'] = winsorize(df['log_gdp_capita'], limits=[0.05, 0.05])
df['log_gdp_wins'] = winsorize(df['log_gdp'], limits=[0.05, 0.05])

dd_ri_p_wins = list()

# log(GDP/capita)
wins1 = PanelOLS.from_formula('log_gdp_capita_wins ~ slave_trade * persistent + TimeEffects', data=df).fit(cov_type="clustered", cluster_entity=True)
dd_ri_p_wins.append(ri_p_values('log_gdp_capita_wins ~ slave_trade * persistent', df, wins1.params, 10000))
print(wins1)

# log(GDP)
wins2 = PanelOLS.from_formula('log_gdp_wins ~ slave_trade * persistent + TimeEffects', data=df).fit(cov_type="clustered", cluster_entity=True)
dd_ri_p_wins.append(ri_p_values('log_gdp_wins ~ slave_trade * persistent', df, wins2.params, 10000))
print(wins2)

# Add controls 
wins3 = PanelOLS.from_formula('log_gdp_capita_wins ~ slave_trade * persistent + {} + TimeEffects'.format(controls), data=df).fit(cov_type="clustered", cluster_entity=True)
dd_ri_p_wins.append(ri_p_values('log_gdp_capita_wins ~ slave_trade * persistent + {}'.format(controls), df, wins3.params, 10000))
print(wins3)

wins4 = PanelOLS.from_formula('log_gdp_wins ~ slave_trade * persistent + {} + TimeEffects'.format(controls), data=df).fit(cov_type="clustered", cluster_entity=True)
dd_ri_p_wins.append(ri_p_values('log_gdp_wins ~ slave_trade * persistent + {}'.format(controls), df, wins4.params, 10000))
print(wins4)

# West Africa 
west_africa = df.loc[df.west_afric == 1, :]
west_africa.set_index(['ethnicity_id', 'year'], inplace=True, drop=False)

wins5 = PanelOLS.from_formula('log_gdp_capita_wins ~ slave_trade * persistent + {} + TimeEffects'.format(controls), data=west_africa).fit(cov_type="clustered", cluster_entity=True)
dd_ri_p_wins.append(ri_p_values('log_gdp_capita_wins ~ slave_trade * persistent + {}'.format(controls), west_africa, wins5.params, 10000))
print(wins5)

wins6 = PanelOLS.from_formula('log_gdp_wins ~ slave_trade * persistent + {} + TimeEffects'.format(controls), data=west_africa).fit(cov_type="clustered", cluster_entity=True)
dd_ri_p_wins.append(ri_p_values('log_gdp_wins ~ slave_trade * persistent + {}'.format(controls), west_africa, wins6.params, 10000))
print(wins6)

to_rename['persistent'] = 'Persistent ethnicity'
to_rename['slave_trade:persistent'] = 'Slave trade x Persistent'
panel_to_tex('tables/diff-in-dif-winsorized.tex', [wins1, wins2, wins3, wins4, wins5, wins6], to_rename, ri=True, ri_p_vals=dd_ri_p_wins)


#%%
# Robustness: continuous measure of persistence 
df['continuous_persistence'] = .01*df['persistenc']
cp_ri_p = list()

cp1 = PanelOLS.from_formula('log_gdp_capita ~ slave_trade * continuous_persistence + TimeEffects', data=df).fit(cov_type="clustered", cluster_entity=True)
cp_ri_p.append(ri_p_values('log_gdp_capita ~ slave_trade * continuous_persistence', df, cp1.params, 10000))
print(cp1)

cp2 = PanelOLS.from_formula('log_gdp ~ slave_trade * continuous_persistence + TimeEffects', data=df).fit(cov_type="clustered", cluster_entity=True)
cp_ri_p.append(ri_p_values('log_gdp ~ slave_trade * continuous_persistence', df, cp2.params, 10000))
print(cp2)

cp3 = PanelOLS.from_formula('log_gdp_capita ~ slave_trade * continuous_persistence + {} + TimeEffects'.format(controls), data=df).fit(cov_type="clustered", cluster_entity=True)
cp_ri_p.append(ri_p_values('log_gdp_capita ~ slave_trade * continuous_persistence + {}'.format(controls), df, cp3.params, 10000))
print(cp3)

cp4 = PanelOLS.from_formula('log_gdp ~ slave_trade * continuous_persistence + {} + TimeEffects'.format(controls), data=df).fit(cov_type="clustered", cluster_entity=True)
cp_ri_p.append(ri_p_values('log_gdp ~ slave_trade * continuous_persistence + {}'.format(controls), df, cp4.params, 10000))
print(cp4)

# West Africa 
west_africa = df.loc[df.west_afric == 1, :]
west_africa.set_index(['ethnicity_id', 'year'], inplace=True, drop=False)

cp5 = PanelOLS.from_formula('log_gdp_capita ~ slave_trade * continuous_persistence + {} + TimeEffects'.format(controls), data=west_africa).fit(cov_type="clustered", cluster_entity=True)
cp_ri_p.append(ri_p_values('log_gdp_capita ~ slave_trade * continuous_persistence + {}'.format(controls), west_africa, cp5.params, 10000))
print(cp5)

cp6 = PanelOLS.from_formula('log_gdp ~ slave_trade * continuous_persistence + {} + TimeEffects'.format(controls), data=west_africa).fit(cov_type="clustered", cluster_entity=True)
cp_ri_p.append(ri_p_values('log_gdp ~ slave_trade * continuous_persistence + {}'.format(controls), west_africa, cp6.params, 10000))
print(cp6)


to_rename['continuous_persistence'] = "Persistence"
to_rename['slave_trade:continuous_persistence'] = 'Slave trade x Persistence'

panel_to_tex('tables/continuous-persistence.tex', [cp1, cp2, cp3, cp4, cp5, cp6], to_rename, ri=True, ri_p_vals=cp_ri_p)


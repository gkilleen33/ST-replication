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
from scipy.stats.mstats import winsorize
from statsmodels.sandbox.regression.gmm import GMM
from sklearn.inspection import permutation_importance

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

# Export results to latex (Note: eliminated stars to adhere to journal submission guidelines)
def assign_stars(coef, pval):
    if pval < .01:
        return '{:,.3f}'.format(coef) # + '\sym{***}'
    elif pval < .05:
        return '{:,.3f}'.format(coef) # + '\sym{**}'
    elif pval < 0.1:
        return '{:,.3f}'.format(coef) # + '\sym{*}'
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
# Create a cross-sectional version of the data with output averaged across years (primary)
# Preserve panel data for appendix 

df_panel = df.copy(deep=True)
df = df.groupby('ethnicity_id').agg('mean')
    
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
        st_reg = smf.ols('{} ~ slave_trade'.format(var), data=df).fit(cov_type='HC0')
        sum_stats.loc[index_mean, 'ST: 1-0'] = assign_stars(st_reg.params['slave_trade'], st_reg.pvalues['slave_trade'])
        sum_stats.loc[index_sd, 'ST: 1-0'] = '({:,.3f})'.format(st_reg.bse['slave_trade'])
        
        sum_stats.loc[index_mean, 'Persistent = 0'] = '{:,.3f}'.format(df[df['persistent'] == 0][var].mean())
        sum_stats.loc[index_sd, 'Persistent = 0'] = '[{:,.3f}]'.format(df[df['persistent'] == 0][var].std())
        st_reg = smf.ols('{} ~ persistent'.format(var), data=df).fit(cov_type='HC0')
        sum_stats.loc[index_mean, 'Persistent: 1-0'] = assign_stars(st_reg.params['persistent'], st_reg.pvalues['persistent'])
        sum_stats.loc[index_sd, 'Persistent: 1-0'] = '({:,.3f})'.format(st_reg.bse['persistent'])
        
        sum_stats.loc[index_blank, 'Variable'] = ''
        
    obs_index = len(sum_stats)
    sum_stats.loc[obs_index, 'Variable'] = 'Observations'
    sum_stats.loc[obs_index, 'Full sample'] = '{:,.0f}'.format(len(df))
    sum_stats.loc[obs_index, 'ST = 0'] = '{:,.0f}'.format(len(df[df['slave_trade'] == 0]))
    sum_stats.loc[obs_index, 'ST: 1-0'] = '{:,.0f}'.format(len(df))
    sum_stats.loc[obs_index, 'Persistent = 0'] = '{:,.0f}'.format(len(df[df['persistent'] == 0]))
    sum_stats.loc[obs_index, 'Persistent: 1-0'] = '{:,.0f}'.format(len(df)) 
    
    dep_vars = ' + '.join(variables)
    f_stat_index = len(sum_stats)
    sum_stats.loc[f_stat_index, 'Variable'] = 'p-val joint orthogonality'
    st_reg = smf.ols('slave_trade ~ {}'.format(dep_vars), data=df).fit(cov_type='HC0')
    sum_stats.loc[f_stat_index, 'ST: 1-0'] = '{:,.3f}'.format(st_reg.f_pvalue)
    st_reg = smf.ols('persistent ~ {}'.format(dep_vars), data=df).fit(cov_type='HC0')
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
                'population': 'Population (mil)'}

df['population'] = (1/10**6)*df['population']

generate_sum_stats('tables/summary.tex', sum_vars, sum_to_rename)
    
#%% 

# log(GDP/capita)
ols1 = smf.ols('log_gdp_capita ~ slave_trade', data=df).fit(cov_type='HC0')
print(ols1.summary())

# log(nightlights)
ols2 = smf.ols('log_gdp ~ slave_trade', data=df).fit(cov_type='HC0')
print(ols2.summary())

# Add controls 
controls = 'diamonds + oil + gold_dep + malaria_pf + malaria_pv + rainfall + neighbors'
ols3 = smf.ols('log_gdp_capita ~ slave_trade + {}'.format(controls), data=df).fit(cov_type='HC0')
print(ols3.summary())

ols4 = smf.ols('log_gdp ~ slave_trade + {}'.format(controls), data=df).fit(cov_type='HC0')
print(ols4.summary())

#%% Define a function to export the PanelOLS results to latex
# Note: Also supports cross-sections

def panel_to_tex(out_path, results, new_var_names=dict(), ri=False, ri_p_vals = None, cross_section=False, rf_bootstrap = None, var_order = None):
    to_export = pd.DataFrame(columns=['Var'])
    var_names = list()
    if var_order:
        var_names = var_order 
    else:
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
panel_to_tex('tables/ols.tex', [ols1, ols2, ols3, ols4], to_rename, cross_section=True)


#%% 
# Now estimate the diff-in-diff model
# log(GDP/capita)

dd_cs_1 = smf.ols('log_gdp_capita ~ slave_trade * persistent', data=df).fit(cov_type='HC0')
print(dd_cs_1.summary())

# log(nightlights)
dd_cs_2 = smf.ols('log_gdp ~ slave_trade * persistent', data=df).fit(cov_type='HC0')
print(dd_cs_2.summary())

# Add controls 
dd_cs_3 = smf.ols('log_gdp_capita ~ slave_trade * persistent + {}'.format(controls), data=df).fit(cov_type='HC0')
print(dd_cs_3.summary())

dd_cs_4 = smf.ols('log_gdp ~ slave_trade * persistent + {}'.format(controls), data=df).fit(cov_type='HC0')
print(dd_cs_4.summary())

# Now restrict to West Africa only 
west_africa_cs = df.loc[df.west_afric == 1, :]
dd_cs_5 = smf.ols('log_gdp_capita ~ slave_trade * persistent + {}'.format(controls), data=west_africa_cs).fit(cov_type='HC0')
print(dd_cs_5.summary())

dd_cs_6 = smf.ols('log_gdp ~ slave_trade * persistent + {}'.format(controls), data=west_africa_cs).fit(cov_type='HC0')
print(dd_cs_6.summary())

to_rename['persistent'] = 'Persistent ethnicity'
to_rename['slave_trade:persistent'] = 'Slave trade x Persistent'
panel_to_tex('tables/diff-in-diff.tex', [dd_cs_1, dd_cs_2, dd_cs_3, dd_cs_4, dd_cs_5, dd_cs_6], to_rename, ri=False, cross_section=True)

#%% 
# Appendix: Diff-in-diff with panel data (to show excluding year fixed effects with night lights doesn't matter)
# log(GDP/capita)
dd1 = PanelOLS.from_formula('log_gdp_capita ~ slave_trade * persistent + TimeEffects', data=df_panel).fit(cov_type="clustered", cluster_entity=True)
print(dd1)

# log(nightlights)
dd2 = PanelOLS.from_formula('log_gdp ~ slave_trade * persistent + TimeEffects', data=df_panel).fit(cov_type="clustered", cluster_entity=True)
print(dd2)

# Add controls 
dd3 = PanelOLS.from_formula('log_gdp_capita ~ slave_trade * persistent + {} + TimeEffects'.format(controls), data=df_panel).fit(cov_type="clustered", cluster_entity=True)
print(dd3)

dd4 = PanelOLS.from_formula('log_gdp ~ slave_trade * persistent + {} + TimeEffects'.format(controls), data=df_panel).fit(cov_type="clustered", cluster_entity=True)
print(dd4)

# Now restrict to West Africa only 
west_africa = df_panel.loc[df_panel.west_afric == 1, :]  # a cutoff by character limit in zonal stats
west_africa.set_index(['ethnicity_id', 'year'], inplace=True, drop=False)

dd5 = PanelOLS.from_formula('log_gdp_capita ~ slave_trade * persistent + {} + TimeEffects'.format(controls), data=west_africa).fit(cov_type="clustered", cluster_entity=True)
print(dd5)

dd6 = PanelOLS.from_formula('log_gdp ~ slave_trade * persistent + {} + TimeEffects'.format(controls), data=west_africa).fit(cov_type="clustered", cluster_entity=True)
print(dd6)

to_rename['persistent'] = 'Persistent ethnicity'
to_rename['slave_trade:persistent'] = 'Slave trade x Persistent'
panel_to_tex('tables/diff-in-diff-panel.tex', [dd1, dd2, dd3, dd4, dd5, dd6], to_rename, ri=False)

#%% 
# Examine the ability to predict persistence of ethnic groups
df['slaves_area'] = df['total_ensl'] / df['area']

pers_controls = controls + ' + num_coloni + Belgium + Britain + France + Germany + Italy + Portugal + Spain'

pers1 = smf.ols('persistent ~ slave_trade', data=df).fit(cov_type='HC0')
print(pers1.summary())

pers2 = smf.ols('persistent ~ slave_trade + {}'.format(pers_controls), data=df).fit(cov_type='HC0')
print(pers2.summary())

pers3 = smf.ols('persistent ~ slaves_area', data=df).fit(cov_type='HC0')
print(pers3.summary())

pers4 = smf.ols('persistent ~ slaves_area + {}'.format(pers_controls), data=df).fit(cov_type='HC0')
print(pers4.summary())

pers5 = smf.ols('persistent ~ slaves_area', data=df[df['slave_trade'] == 1]).fit(cov_type='HC0')
print(pers5.summary())

pers6 = smf.ols('persistent ~ slaves_area + {}'.format(pers_controls), df[df['slave_trade'] == 1]).fit(cov_type='HC0')
print(pers6.summary())

order_pers = ['Intercept', 'slave_trade', 'slaves_area', 'diamonds', 'oil', 'gold_dep', 
                         'malaria_pf', 'malaria_pv', 'rainfall', 'neighbors',
                         'num_coloni', 'Belgium', 'Britain', 'France',
                         'Germany', 'Italy', 'Portugal', 'Spain']

to_rename_pers = {'slaves_area': 'Slaves/area', 'slave_trade': 'Slave trade',
                           'diamonds': 'Diamond deposits', 
                           'oil': 'Oil', 'gold_dep': 'Gold deposits', 
                           'malaria_pf': 'Malaria (Pf)', 'malaria_pv': 'Malaria (Pv)', 
                           'rainfall': 'Annual rainfall (mm/day)', 'neighbors': 'Neighbors',
                           'num_coloni': 'Colonizers (number)', 'Belgium': 'Belgium (colonized by)'}

panel_to_tex('tables/persistence.tex', [pers1, pers2, pers3, pers4, pers5, pers6], to_rename_pers, 
             ri=False, cross_section=True, var_order = order_pers)


#%% 
# For appendix: regression on raw nightlights 

# log(NL/capita)
nl1 = PanelOLS.from_formula('log_nightlights_capita ~ slave_trade * persistent + TimeEffects', data=df_panel).fit(cov_type="clustered", cluster_entity=True)
print(nl1)

# log(nightlights)
nl2 = PanelOLS.from_formula('log_nightlights ~ slave_trade * persistent + TimeEffects', data=df_panel).fit(cov_type="clustered", cluster_entity=True)
print(nl2)

# Add controls 
nl3 = PanelOLS.from_formula('log_nightlights_capita ~ slave_trade * persistent + {} + TimeEffects'.format(controls), data=df_panel).fit(cov_type="clustered", cluster_entity=True)
print(nl3)

nl4 = PanelOLS.from_formula('log_nightlights ~ slave_trade * persistent + {} + TimeEffects'.format(controls), data=df_panel).fit(cov_type="clustered", cluster_entity=True)
print(nl4)

# Now restrict to West Africa only 
nl5 = PanelOLS.from_formula('log_nightlights_capita ~ slave_trade * persistent + {} + TimeEffects'.format(controls), data=west_africa).fit(cov_type="clustered", cluster_entity=True)
print(nl5)

nl6 = PanelOLS.from_formula('log_nightlights ~ slave_trade * persistent + {} + TimeEffects'.format(controls), data=west_africa).fit(cov_type="clustered", cluster_entity=True)
print(nl6)

to_rename['persistent'] = 'Persistent ethnicity'
to_rename['slave_trade:persistent'] = 'Slave trade x Persistent'
panel_to_tex('tables/diff-in-diff-nl.tex', [nl1, nl2, nl3, nl4, nl5, nl6], to_rename, ri=False)

#%% 

from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


# Now attempt to predict persistence with a random forest among groups that did not participate in the slave trade
df_st0 = df[df['slave_trade'] == 0]

# First use all observations for hyperparameter searching using CV
X = df_st0[['diamonds', 'oil', 'gold_dep', 'malaria_pf', 'malaria_pv', 'rainfall', 'surface',
           'num_coloni', 'lat', 'lon', 'Belgium', 'Britain', 'France', 'Germany', 'Italy', 'Portugal', 'Spain']]
features = list(X.columns)
X = X.values

y = df_st0[['persistent']].values.ravel()

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
    'criterion': ['gini'],
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
                        scoring = 'neg_log_loss', cv = 10, 
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
df_rf.sort_values(by=['slave_trade'], inplace=True)
df_rf['row_by_st'] = df_rf.groupby(['slave_trade']).cumcount() + 1
df_rf.loc[df_rf['slave_trade'] == 0, 'fold'] = pd.cut(df_rf[df_rf['slave_trade'] == 0]['row_by_st'], bins=5, labels=False)
df_rf.loc[df_rf['slave_trade'] == 1, 'fold'] = pd.cut(df_rf[df_rf['slave_trade'] == 1]['row_by_st'], bins=5, labels=False)

# Now randomly reshuffle the folds within slave trade status, keeping groups fixed
import random
df_rf['ethnicity_id'] = df_rf.index
ids_1 = df_rf.loc[df_rf['slave_trade'] == 0, 'ethnicity_id'].unique()
random.Random(0).shuffle(ids_1)

ids_2 = df_rf.loc[df_rf['slave_trade'] == 1, 'ethnicity_id'].unique()
random.Random(0).shuffle(ids_2)

ids = list(ids_1) + list(ids_2)  # So shuffling occurs by group and within slave trade status 

temp = df_rf.copy(deep=True)
temp = temp.set_index('ethnicity_id').loc[ids].reset_index()

df_rf['fold'] = list(temp['fold'])

results = df_rf.copy(deep=True)
results = results.loc[:, ['ethnicity_id', 'fold', 'persistent', 'slave_trade']]
results['predicted_persistence'] = None
results['predicted_probabilities'] = None

# Now train, predict, and score using 5-fold CV
feature_importance = list() 
for k in range(5):
    df_k = df_rf.loc[df_rf['fold'] == k]
    df_nk = df_rf.loc[df_rf['fold'] != k]
    
    # Extract the observations with slave_trade = 0 in groups nk for training 
    X_nk = df_nk[df_nk['slave_trade'] == 0]
    X_nk = X_nk[['diamonds', 'oil', 'gold_dep', 'malaria_pf', 'malaria_pv', 'rainfall', 'surface',
                 'num_coloni', 'lat', 'lon', 'Belgium', 'Britain', 'France', 'Germany', 
                 'Italy', 'Portugal', 'Spain']]
    y_nk = df_nk[df_nk['slave_trade'] == 0]
    y_nk = y_nk[['persistent']].values.ravel()

    # Train model 
    rf.fit(X_nk, y_nk)
    
    # Now predict on fold k 
    X_k = df_k[['diamonds', 'oil', 'gold_dep', 'malaria_pf', 'malaria_pv', 'rainfall', 'surface',
                'num_coloni', 'lat', 'lon', 'Belgium', 'Britain', 'France', 
                'Germany', 'Italy', 'Portugal', 'Spain']].values
    results.loc[results['fold'] == k, 'predicted_persistence'] = rf.predict(X_k)
    results.loc[results['fold'] == k, 'predicted_probabilities'] = rf.predict_proba(X_k)[:, 1]
    result = permutation_importance(
        rf, X_nk, y_nk, n_repeats=10, random_state=0, n_jobs=-1
    )
    feature_importance.append(result)

results['predicted_persistence'] = results['predicted_persistence'].astype('int64')

average_importance = 0.2*feature_importance[0].importances_mean + 0.2*feature_importance[1].importances_mean + 0.2*feature_importance[2].importances_mean + 0.2*feature_importance[3].importances_mean + 0.2*feature_importance[4].importances_mean
importance = np.hstack([feature_importance[0].importances, feature_importance[1].importances, feature_importance[2].importances, feature_importance[3].importances, feature_importance[4].importances])

feature_names = np.array(['Diamond deposits', 'Oil', 'Gold deposits', 'Malaria (Pf)', 'Malaria (Pv)', 
                 'Rainfall', 'Land type', 'Num colonizers', 
                 'Latitude', 'Longitude',
                 'Belgium', 'Britain', 'France', 
                 'Germany', 'Italy', 'Portugal', 'Spain'])

sorted_idx = average_importance.argsort()
fig, ax = plt.subplots()
ax.boxplot(
    importance[sorted_idx].T, vert=False, labels=feature_names[sorted_idx]
)
ax.set_title("Permutation Importances")
fig.savefig('figures/feature_importance.png', bbox_inches = 'tight', dpi=200)
fig.show()


evaluate_model(results.loc[results['slave_trade'] == 0, 'predicted_persistence'], 
               results.loc[results['slave_trade'] == 0, 'predicted_probabilities'], 
               results.loc[results['slave_trade'] == 0, 'persistent'])

plt.savefig('figures/roc.png', bbox_inches = 'tight', dpi=200)
plt.show()

#%% Now use predicted persistence as a proxy for persistence, with GMM estimation according to Botosaru and Gutierrez 

df = df.merge(results[['predicted_probabilities']], on=['ethnicity_id'])
df['predicted_probabilities'] = df['predicted_probabilities'].astype(float)

class proxyDiD(GMM):
    def __init__(self, *args, **kwds):
        # Set appropriate counts for moment conditions and parameters
        kwds.setdefault('k_moms', 6)
        kwds.setdefault('k_params', 6)
        super(proxyDiD, self).__init__(*args, **kwds)


    def momcond(self, params):
        psi0, psi1, gamma, delta0, delta1, kappa = params
        y = self.endog
        slave_trade = self.exog[:,0]
        persistent = self.exog[:,1]
        predicted_persistence = self.instrument
        m1 = (1 - slave_trade)*predicted_persistence*((persistent - gamma*predicted_persistence)/(gamma*predicted_persistence*(1-gamma*predicted_persistence)))
        m2 = slave_trade*(y-delta1-psi1*gamma*predicted_persistence-predicted_persistence*kappa)
        m3 = slave_trade*predicted_persistence*(y-delta1-psi1*gamma*predicted_persistence-predicted_persistence*kappa)
        m4 = (1 - slave_trade)*(y - delta0 - psi0*persistent - predicted_persistence*kappa)
        m5 = (1 - slave_trade)*predicted_persistence*(y - delta0 - psi0*persistent - predicted_persistence*kappa)
        m6 = (1 - slave_trade)*persistent*(y - delta0 - psi0*persistent - predicted_persistence*kappa)
        g = np.column_stack((m1, m2, m3, m4, m5, m6))
        return g

class proxyDiDControls(GMM):
    def __init__(self, *args, **kwds):
        # Set appropriate counts for moment conditions and parameters
        kwds.setdefault('k_moms', 14)
        kwds.setdefault('k_params', 14)
        super(proxyDiDControls, self).__init__(*args, **kwds)


    def momcond(self, params):
        psi0, psi1, gamma, delta0, delta1, kappa, c1, c2, c3, c4, d1, d2, d3, d4 = params
        y = self.endog
        slave_trade = self.exog[:,0]
        persistent = self.exog[:,1]

        predicted_persistence = self.instrument
        
        control_terms = c1*self.exog[:,2] + c2*self.exog[:,3] + c3*self.exog[:,4] + c4*self.exog[:,5] 
        control_terms_st1 = d1*self.exog[:,2] + d2*self.exog[:,3] + d3*self.exog[:,4] + d4*self.exog[:,5] 
        
        m1 = (1 - slave_trade)*predicted_persistence*((persistent - gamma*predicted_persistence)/(gamma*predicted_persistence*(1-gamma*predicted_persistence)))
        m2 = slave_trade*(y-delta1-psi1*gamma*predicted_persistence-predicted_persistence*kappa - control_terms_st1)
        m3 = slave_trade*predicted_persistence*(y-delta1-psi1*gamma*predicted_persistence-predicted_persistence*kappa - control_terms_st1)
        m4 = slave_trade*self.exog[:,2]*(y-delta1-psi1*gamma*predicted_persistence-predicted_persistence*kappa - control_terms_st1)
        m5 = slave_trade*self.exog[:,3]*(y-delta1-psi1*gamma*predicted_persistence-predicted_persistence*kappa - control_terms_st1)
        m6 = slave_trade*self.exog[:,4]*(y-delta1-psi1*gamma*predicted_persistence-predicted_persistence*kappa - control_terms_st1)
        m7 = slave_trade*self.exog[:,5]*(y-delta1-psi1*gamma*predicted_persistence-predicted_persistence*kappa - control_terms_st1)
        m8 = (1 - slave_trade)*(y - delta0 - psi0*persistent - predicted_persistence*kappa - control_terms)
        m9 = (1 - slave_trade)*predicted_persistence*(y - delta0 - psi0*persistent - predicted_persistence*kappa - control_terms)
        m10 = (1 - slave_trade)*persistent*(y - delta0 - psi0*persistent - predicted_persistence*kappa - control_terms)
        m11 = (1 - slave_trade)*self.exog[:,2]*(y - delta0 - psi0*persistent - predicted_persistence*kappa - control_terms)
        m12 = (1 - slave_trade)*self.exog[:,3]*(y - delta0 - psi0*persistent - predicted_persistence*kappa - control_terms)
        m13 = (1 - slave_trade)*self.exog[:,4]*(y - delta0 - psi0*persistent - predicted_persistence*kappa - control_terms)
        m14 = (1 - slave_trade)*self.exog[:,5]*(y - delta0 - psi0*persistent - predicted_persistence*kappa - control_terms)
        g = np.column_stack((m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14))
        return g

beta0 = np.ones(6)
gmm_results = proxyDiD(endog=df['log_gdp_capita'], 
                       exog=df[['slave_trade', 'persistent']], 
                       instrument=df['predicted_probabilities']).fit(beta0, maxiter = 1)  # Just identified so just want one-step
print(gmm_results.summary())

gmm_results2 = proxyDiD(endog=df['log_gdp'], 
                       exog=df[['slave_trade', 'persistent']], 
                       instrument=df['predicted_probabilities']).fit(beta0, maxiter = 1)
print(gmm_results2.summary())

beta1 = np.ones(14)
gmm_results3 = proxyDiDControls(endog=df['log_gdp_capita'], 
                       exog=df[['slave_trade', 'persistent', 'diamonds', 'oil', 'gold_dep',
                                'neighbors']], 
                       instrument=df['predicted_probabilities']).fit(beta1, maxiter = 1)
print(gmm_results3.summary())
    
gmm_results4 = proxyDiDControls(endog=df['log_gdp'], 
                       exog=df[['slave_trade', 'persistent', 'diamonds', 'oil', 'gold_dep',
                                'neighbors']], 
                       instrument=df['predicted_probabilities']).fit(beta1, maxiter = 1)
print(gmm_results4.summary())

west_africa = df.loc[df.west_afric == 1, :]

gmm_results5 = proxyDiDControls(endog=west_africa['log_gdp_capita'], 
                       exog=west_africa[['slave_trade', 'persistent', 'diamonds', 'oil', 'gold_dep',
                                'neighbors']], 
                       instrument=west_africa['predicted_probabilities']).fit(beta1, maxiter = 1)
print(gmm_results5.summary())
    
gmm_results6 = proxyDiDControls(endog=west_africa['log_gdp'], 
                       exog=west_africa[['slave_trade', 'persistent', 'diamonds', 'oil', 'gold_dep',
                                'neighbors']], 
                       instrument=west_africa['predicted_probabilities']).fit(beta1, maxiter = 1)
print(gmm_results6.summary())

#%% 
# Now define function to write the GMM results to LaTex
# Need to calculate the comparable results (e.g. diff-in-diff coefficient) with post-estimation tests
def gmm_to_tex(out_path, results):
    # T-test for diff-in-diff parameter
    diff_in_diff_coefs = list()
    diff_in_diff_ses = list()
    diff_in_diff_p_vals = list()
    for result in results:
        if len(result.params) == 6:
            att_t_test = result.t_test(np.array([[-1, 1, 0, 0, 0, 0]])).summary_frame()
        else:
            att_t_test = result.t_test(np.array([[-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])).summary_frame()
        att = att_t_test['coef'].iloc[0]
        att_se = att_t_test['std err'].iloc[0]
        att_p = att_t_test['P>|z|'].iloc[0]
        diff_in_diff_coefs.append(att)
        diff_in_diff_ses.append(att_se)
        diff_in_diff_p_vals.append(att_p)
        
    # T-test for slave trade parameter
    st_coefs = list()
    st_ses = list()
    st_p_vals = list()
    for result in results:
        if len(result.params) == 6:
            st_t_test = result.t_test(np.array([[0, 0, 0, -1, 1, 0]])).summary_frame()
        else:
            st_t_test = result.t_test(np.array([[0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])).summary_frame()
        st = st_t_test['coef'].iloc[0]
        st_se = st_t_test['std err'].iloc[0]
        st_p = st_t_test['P>|z|'].iloc[0]
        st_coefs.append(st)
        st_ses.append(st_se)
        st_p_vals.append(st_p)
        
    to_export = pd.DataFrame(columns=['Var'])
    var_names = ['Intercept', 'Slave trade', 'Persistent ethnicity', 'Slave trade x Persistent']
    
    for v in var_names:
        to_export.loc[len(to_export)] = v
        to_export.loc[len(to_export)] = ''  # For SE
        to_export.loc[len(to_export)] = ''  # Blank line between variables for display
    
    to_export.loc[len(to_export)] = 'Proxy t-stat'
    to_export.loc[len(to_export)] = 'Controls'
    to_export.loc[len(to_export)] = 'Observations'
    
    # We have now populated the labels for each variable. Now iterate over the models and populate the results
    i = 1 
    for r in results:
        to_export['({})'.format(i)] = ''  # Column to store results
        
        # Intercept (delta0)
        p_index = to_export[to_export['Var'] == 'Intercept'].index[0]
        to_export.loc[p_index, '({})'.format(i)] = assign_stars(r.params[3], r.pvalues[3])  
        to_export.loc[p_index + 1, '({})'.format(i)] = '({:,.3f})'.format(r.bse[3])
        
        # Slave trade (delta1 - delta0)
        p_index = to_export[to_export['Var'] == 'Slave trade'].index[0]
        to_export.loc[p_index, '({})'.format(i)] = assign_stars(st_coefs[i-1], st_p_vals[i-1])  
        to_export.loc[p_index + 1, '({})'.format(i)] = '({:,.3f})'.format(st_ses[i-1])
        
        # Persistence (Psi0)
        p_index = to_export[to_export['Var'] == 'Persistent ethnicity'].index[0]
        to_export.loc[p_index, '({})'.format(i)] = assign_stars(r.params[0], r.pvalues[0])  
        to_export.loc[p_index + 1, '({})'.format(i)] = '({:,.3f})'.format(r.bse[0])
        
        # Slave trade x persistent 
        p_index = to_export[to_export['Var'] == 'Slave trade x Persistent'].index[0]
        to_export.loc[p_index, '({})'.format(i)] = assign_stars(diff_in_diff_coefs[i-1], diff_in_diff_p_vals[i-1])  
        to_export.loc[p_index + 1, '({})'.format(i)] = '({:,.3f})'.format(diff_in_diff_ses[i-1])
        
        to_export.loc[(to_export['Var'] == 'Proxy t-stat'), '({})'.format(i)] = '{:,.3f}'.format(r.tvalues[2])
        
        # Now controls if included 
        if len(r.params) > 6:
            to_export.loc[(to_export['Var'] == 'Controls'), '({})'.format(i)] = 'Yes'
        else:
            to_export.loc[(to_export['Var'] == 'Controls'), '({})'.format(i)] = 'No'
        
        to_export.loc[(to_export['Var'] == 'Observations'), '({})'.format(i)] = '{:,.0f}'.format(r.nobs)
        
        i += 1
    with open(out_path, 'w') as f:
        to_export.to_latex(f, header=False, index=False, na_rep='', escape=False)
    
# Now output the results 
gmm_to_tex('tables/diff-in-diff-pred.tex', [gmm_results, gmm_results2, gmm_results3, gmm_results4, gmm_results5, gmm_results6])    
    

#%% 
# Winsorized diff in diff 
df['log_gdp_capita_wins'] = winsorize(df['log_gdp_capita'], limits=[0.05, 0.05])
df['log_gdp_wins'] = winsorize(df['log_gdp'], limits=[0.05, 0.05])


# log(GDP/capita)
wins1 = smf.ols('log_gdp_capita_wins ~ slave_trade * persistent', data=df).fit(cov_type='HC0')
print(wins1.summary())

# log(GDP)
wins2 = smf.ols('log_gdp_wins ~ slave_trade * persistent', data=df).fit(cov_type='HC0')
print(wins2.summary())

# Add controls 
wins3 = smf.ols('log_gdp_capita_wins ~ slave_trade * persistent + {}'.format(controls), data=df).fit(cov_type='HC0')
print(wins3.summary())

wins4 = smf.ols('log_gdp_wins ~ slave_trade * persistent + {}'.format(controls), data=df).fit(cov_type='HC0')
print(wins4.summary())

# West Africa 
west_africa = df.loc[df.west_afric == 1, :]

wins5 = smf.ols('log_gdp_capita_wins ~ slave_trade * persistent + {}'.format(controls), data=west_africa).fit(cov_type='HC0')
print(wins5.summary())

wins6 = smf.ols('log_gdp_wins ~ slave_trade * persistent + {}'.format(controls), data=west_africa).fit(cov_type='HC0')
print(wins6.summary())

to_rename['persistent'] = 'Persistent ethnicity'
to_rename['slave_trade:persistent'] = 'Slave trade x Persistent'
panel_to_tex('tables/diff-in-dif-winsorized.tex', [wins1, wins2, wins3, wins4, wins5, wins6], to_rename, ri=False, cross_section=True)


#%%
# Robustness: continuous measure of persistence 
df['continuous_persistence'] = .01*df['persistenc']

cp1 = smf.ols('log_gdp_capita ~ slave_trade * continuous_persistence', data=df).fit(cov_type='HC0')
print(cp1.summary())

cp2 = smf.ols('log_gdp ~ slave_trade * continuous_persistence', data=df).fit(cov_type='HC0')
print(cp2.summary())

cp3 = smf.ols('log_gdp_capita ~ slave_trade * continuous_persistence + {}'.format(controls), data=df).fit(cov_type='HC0')
print(cp3.summary())

cp4 = smf.ols('log_gdp ~ slave_trade * continuous_persistence + {}'.format(controls), data=df).fit(cov_type='HC0')
print(cp4.summary())

# West Africa 
west_africa = df.loc[df.west_afric == 1, :]

cp5 = smf.ols('log_gdp_capita ~ slave_trade * continuous_persistence + {}'.format(controls), data=west_africa).fit(cov_type='HC0')
print(cp5.summary())

cp6 = smf.ols('log_gdp ~ slave_trade * continuous_persistence + {}'.format(controls), data=west_africa).fit(cov_type='HC0')
print(cp6.summary())


to_rename['continuous_persistence'] = "Persistence"
to_rename['slave_trade:continuous_persistence'] = 'Slave trade x Persistence'

panel_to_tex('tables/continuous-persistence.tex', [cp1, cp2, cp3, cp4, cp5, cp6], to_rename, ri=False, cross_section=True)

#%%
# Robustness: ST = 0 if slave exported very low (fewer than 1,000 slaves exported)
df['slave_trade'] = np.where(df['total_ensl'] > 0.1, 1, 0)  # Call it slave_trade still so labeling still works

dd_cons_1 = smf.ols('log_gdp_capita ~ slave_trade * persistent', data=df).fit(cov_type='HC0')
print(dd_cons_1.summary())

# log(nightlights)
dd_cons_2 = smf.ols('log_gdp ~ slave_trade * persistent', data=df).fit(cov_type='HC0')
print(dd_cons_2.summary())

# Add controls 
dd_cons_3 = smf.ols('log_gdp_capita ~ slave_trade * persistent + {}'.format(controls), data=df).fit(cov_type='HC0')
print(dd_cons_3.summary())

dd_cons_4 = smf.ols('log_gdp ~ slave_trade * persistent + {}'.format(controls), data=df).fit(cov_type='HC0')
print(dd_cons_4.summary())

# Now restrict to West Africa only 
west_africa_cs = df.loc[df.west_afric == 1, :]
dd_cons_5 = smf.ols('log_gdp_capita ~ slave_trade * persistent + {}'.format(controls), data=west_africa_cs).fit(cov_type='HC0')
print(dd_cons_5.summary())

dd_cons_6 = smf.ols('log_gdp ~ slave_trade * persistent + {}'.format(controls), data=west_africa_cs).fit(cov_type='HC0')
print(dd_cons_6.summary())

to_rename['persistent'] = 'Persistent ethnicity'
to_rename['slave_trade:persistent'] = 'Slave trade x Persistent'
panel_to_tex('tables/diff-in-diff-conservative.tex', [dd_cons_1, dd_cons_2, dd_cons_3, dd_cons_4, dd_cons_5, dd_cons_6], to_rename, ri=False, cross_section=True)

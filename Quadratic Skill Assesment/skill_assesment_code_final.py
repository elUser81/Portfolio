#!/usr/bin/env python
# coding: utf-8

# In[260]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import scipy

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[261]:


import plotly.graph_objs as go 
import chart_studio.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) 


# In[262]:


donations = pd.read_csv('skills_assessment_data - donation_table.csv')
donors = pd.read_csv('skills_assessment_data - donor_table.csv')
media = pd.read_csv('skills_assessment_data - media_table.csv')
donations.head()


# In[263]:


donors.head()


# In[264]:


media.head()


# In[265]:


data = donors
data['channel'], data['campaign_name'] = media['channel'], media['campaign_name']
data['donation_revenue'] = donations['donation_revenue']
data.head()


# In[266]:


data.groupby('state').sum()


# In[267]:


data['channel'].unique()


# In[268]:


data = data.replace(to_replace = {'e-mail':'email', 'sarch':'search', 'org':'organic'})


# In[269]:


totals_by_state = data.groupby('state').sum().drop(columns = ['donation_id', 'age'])
means_by_state = data.groupby('state').mean().drop(columns = ['donation_id', 'age'])


# In[270]:


num_donations_by_state = data.groupby('state').size()


# In[301]:


total_rev = data.sum()['donation_revenue']
rev_by_state = data.groupby('state').sum()['donation_revenue']
pct_rev = (rev_by_state/total_rev)*100


# In[306]:


map_data = dict(type = 'choropleth',
            locations = totals_by_state.index.values,
            locationmode = 'USA-states',
            colorscale= 'Portland',
            z=pct_rev,
            colorbar = {'title':'Percent of Total Revenue'})

layout = dict(geo = {'scope':'usa'})


# ## Precent of Total Revenue By State

# In[307]:


choromap = go.Figure(data = [map_data],layout = layout)
iplot(choromap)


# In[ ]:





# In[273]:


num_general_by_state = data[data['campaign_name'] == 'General'].groupby('state').count()['age']
num_homelessness_by_state = data[data['campaign_name'] == 'Homelessness'].groupby('state').count()['age']
num_poverty_by_state = data[data['campaign_name'] == 'Poverty'].groupby('state').count()['age']


# In[274]:


num_campaign_by_state = pd.DataFrame()
num_campaign_by_state['General'] = num_general_by_state
num_campaign_by_state['Homelessness'] = num_homelessness_by_state
num_campaign_by_state['Poverty'] = num_poverty_by_state


# In[275]:


num_campaign_by_state.plot.bar(title = 'Number of Donations by State and Campaign Name')


# In[276]:


recurring_by_state = data[data['donor_type'] == 'recurring'].groupby('state').count()['age']
total_donations_by_state = data.groupby('state').count()['age']
pct_recurring =  (recurring_by_state/total_donations_by_state)*100


# In[277]:


map_data = dict(type = 'choropleth',
            locations = totals_by_state.index.values,
            locationmode = 'USA-states',
            colorscale= 'Portland',
            z=pct_recurring,
            colorbar = {'title':'Percent of Recurring Donations'})

layout = dict(geo = {'scope':'usa'})

choromap = go.Figure(data = [map_data],layout = layout)
iplot(choromap)


# In[278]:


total_recurring_rev = data[data['donor_type'] == 'recurring'].groupby('state').sum()['donation_revenue']
total_rev_by_state = data.groupby('state').sum()['donation_revenue']

pct_recurring_rev_by_state = (total_recurring_rev/total_rev_by_state)*100

map_data = dict(type = 'choropleth',
            locations = totals_by_state.index.values,
            locationmode = 'USA-states',
            colorscale= 'Portland',
            z=pct_recurring_rev_by_state,
            colorbar = {'title':'Percent Revenue from Recurring Donations'})

layout = dict(geo = {'scope':'usa'})

choromap = go.Figure(data = [map_data],layout = layout)
iplot(choromap)


# In[ ]:





# In[279]:


data.head()


# In[334]:


one_time = data[data['donor_type'] == 'one-time'].groupby('state').count()['age']




# In[321]:


display_by_state = data[data['channel'] == 'display'].groupby('state').count()['age']
email_by_state = data[data['channel'] == 'email'].groupby('state').count()['age']
organic_by_state = data[data['channel'] == 'organic'].groupby('state').count()['age']
search_by_state = data[data['channel'] == 'search'].groupby('state').count()['age']
total = data.groupby('state').count()['age']

pct_display = (display_by_state/total)*100
pct_email = (email_by_state/total)*100
pct_organic = (organic_by_state/total)*100
pct_search = (search_by_state/total)*100
d = {'pct_display': pct_display, 'pct_email': pct_email, 'pct_organic':pct_organic, 'pct_search':pct_search,'total_donations': total}
channel_pcts = pd.DataFrame(data = d)
sns.heatmap(channel_pcts.drop(columns = ['total_donations']))
channel_pcts

total


# In[330]:


data.head()
sums = data.groupby('state').sum()
total_spending = sums['spending_on_food'] + sums['entertainment']
pct_food = sums['spending_on_food']/total_spending
pct_ent = sums['entertainment']/total_spending
pct_ent


# In[282]:


rec_donations = data[data['donor_type'] == 'recurring']
onetime_donations = data[data['donor_type'] == 'one-time']

total = rec_donations.count()['age']

pct_recs_by_channel = (rec_donations.groupby('channel').count()['age']/total)*100
pct_recs_by_channel


# In[283]:


means = data.groupby('state').mean().drop(columns = ['donation_id'])
means_and_pct_by_channel = pd.concat([means, channel_pcts], axis = 1)

#means_and_pct_by_channel
#sns.heatmap(means_and_pct_by_channel.corr(),cmap='coolwarm')
#sns.jointplot(x = 'spending_on_food', y = 'donation_revenue', data = data, kind = 'reg')


# In[284]:


recurring_by_campaign = data[data['donor_type'] == 'recurring'].groupby('campaign_name').count()['age']
total_by_campaign = data.groupby('campaign_name').count()['age']
pct_recurring_by_campaign = (recurring_by_campaign/total_by_campaign)*100
pct_recurring_by_campaign


# In[285]:


reven_desc = data.sort_values(by = ['donation_revenue'], ascending = False)

reven_desc[reven_desc['donor_type'] == 'recurring']


# In[313]:


revenue_poverty = data[data['campaign_name'] == 'Poverty'].groupby('state').sum()['donation_revenue']
revenue_homelessness = data[data['campaign_name'] == 'Homelessness'].groupby('state').sum()['donation_revenue']
revenue_general = data[data['campaign_name'] == 'General'].groupby('state').sum()['donation_revenue']
total_rev = data.groupby('state').sum()['donation_revenue']

pct_rev_poverty = (revenue_poverty/total_rev)*100
pct_rev_homelessness = (revenue_homelessness/total_rev)*100
pct_rev_general = (revenue_general/total_rev)*100

d = {'pct_display': pct_display, 'pct_email': pct_email, 'pct_organic':pct_organic, 'pct_search':pct_search, 
     'pct_recurring': pct_recurring, 'total_donations': total}


d = {'pct_rev_poverty':pct_rev_poverty, 'pct_rev_homelessness':pct_rev_homelessness, 'pct_rev_general':pct_rev_general}

pct_revs = pd.DataFrame(data = d)
pct_revs.plot.bar(title = 'Percent of State Revenue by State and Campaign')


# In[287]:


total_rev_by_campaign = data.groupby('campaign_name').sum()['donation_revenue']
total_rev = data.sum()['donation_revenue']

pct_rev_totals = (total_rev_by_campaign/total_rev)*100
pct_rev_totals


# In[288]:


data.groupby('campaign_name').sum().drop(columns = ['donation_id', 'age']).plot.bar()


# In[335]:


data.groupby('channel').sum().drop(columns = ['donation_id', 'age']).plot.bar()


# In[289]:


data.groupby('campaign_name').count()['age']


# In[290]:


data.drop(columns = ['donation_id', 'age']).groupby(['state','channel']).sum()['donation_revenue'].plot.bar()


# In[291]:


data.groupby(['campaign_name','channel']).count()['age'].plot.bar(title = 'Number of Donations by Campaign Name and Channel')


# In[299]:


state_rev = data.groupby('state').sum()['donation_revenue']
num_donations = data.groupby('state').count()['donation_revenue']
d = {'num_donations': num_donations, 'state_revenue':state_rev}
df = pd.DataFrame(data = d)
sns.jointplot(data = df, x = 'num_donations', y = 'state_revenue', kind = 'reg')
df.corr()


# In[415]:


sns.lmplot(data = data, x = 'age', y = 'donation_revenue', hue = 'campaign_name')


# In[420]:


g = sns.FacetGrid(data, col = 'state')
g.map(plt.hist, 'donation_revenue')


# In[431]:


data.groupby('state').mean()[['spending_on_food', 'entertainment']].plot.bar(title = 'Average Spending by State')


# In[365]:


data.groupby('campaign_name').mean()[['spending_on_food', 'entertainment']].plot.bar()


# In[363]:


data.groupby('channel').sum()[['spending_on_food', 'entertainment']].plot.bar()


# In[396]:


data['total_spending'] = data['spending_on_food'] + data['entertainment']
data['pct_rev_of_spending'] = (data['donation_revenue']/data['total_spending'])*100
sns.jointplot(data = data, x = 'total_spending', y = 'donation_revenue' )


# In[404]:


sns.lmplot(data = data[data.donor_type != ' '], x = 'spending_on_food', y = 'donation_revenue', hue = 'donor_type' )


# In[398]:


sns.lmplot(data = data, x = 'spending_on_food', y = 'donation_revenue', hue = 'gender' )


# In[399]:


sns.lmplot(data = data, x = 'spending_on_food', y = 'donation_revenue', hue = 'channel' )


# In[400]:


data.head()


# In[401]:


data.sort_values(by = ['pct_rev_of_spending'], ascending = False).mean()


# In[389]:


d = data[['pct_rev_of_spending', 'state']]
d = d[(d.pct_rev_of_spending <=500) & (d.pct_rev_of_spending >= 0) ]
d = d[d['state'] == 'SD']
sns.distplot(d['pct_rev_of_spending'])


# In[437]:


d = data#[['pct_rev_of_spending', 'state']]
d = d[(d.pct_rev_of_spending <=500) & (d.pct_rev_of_spending >= 0) ]
#g = sns.FacetGrid(d, col = 'state')
#g.map(plt.hist, 'pct_rev_of_spending')

sns.lmplot(data = d, x = 'age', y = 'donation_revenue', hue = 'donor_type' )


# In[435]:


state_revs = data.groupby('state').mean()['donation_revenue']
state_spending = data.groupby('state').mean()['total_spending']

pct_rev_state_spending = (state_revs/state_spending)*100

sns.distplot(pct_rev_state_spending)
pct_rev_state_spending.plot.bar(title = 'Average Donation Revenue as a Percent of Total Spending')

pct_rev_state_spending


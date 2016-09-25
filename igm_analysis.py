# -*- coding: utf-8 -*-

import os
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import dateutil.parser as dtutil
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib
from functools import partial
from collections import OrderedDict
from bs4 import BeautifulSoup

if __name__ == '__main__':

    ############
    # SCRAPING #
    ############
    
    if not os.path.exists('igmchicago.pkl'):
    # Get detail page links
        domain = 'http://www.igmchicago.org'
        response = requests.get(domain + '/igm-economic-experts-panel')
        soup = BeautifulSoup(response.text, 'lxml')
        polls = soup.findAll('div', {'class': 'poll-listing'})
        topics = {}
        for poll in polls:
            topic = poll.find('h2').get_text()
            handle = poll.find('h3', {'class': 'surveyQuestion'}).a['href']
            topics[topic] = handle
        
        # Scrape detail pages
        data = []
        for topic, handle in topics.items():
            print(handle)
            response = requests.get(domain + handle)
            soup = BeautifulSoup(response.text, 'lxml')
            survey_questions = list('ABCD')[:len(soup.findAll('h3', {'class': 'surveyQuestion'}))]
            survey_date = soup.find('h6').get_text()
            tables = soup.findAll('table', {'class': 'responseDetail'})
            for survey_question, table in zip(survey_questions, tables):
                rows = table.findAll('tr')#, {'class': 'parent-row'})
                for row in rows:
                    if row.get('class') == ['parent-row']:
                        cells = row.findAll('td')
                        response = cells[2].get_text().strip()
                        confidence = cells[3].get_text().strip()
                        comment = cells[4].get_text().strip()
                        tmp_data = {
                            'survey_date': dtutil.parse(survey_date),
                            'topic_name': topic,
                            'topic_url': domain + handle,
                            'survey_question': survey_question,
                            'economist_name': cells[0].get_text().strip(),
                            'economist_url':  domain + cells[0].a['href'],
                            'economist_headshot': domain + cells[0].img['src'],
                            'institution': cells[1].get_text().strip(),
                            'response': response,
                            'confidence': confidence,
                            'comment': comment,
                        }
                        
                        # If response, confidence and comment are all '---', this is a newly added economist
                        # Update the dictionary with the next row's information
                        if all([x == '---' for x in (response, confidence, comment)]):
                            nextRow = row.nextSibling.nextSibling
                            if nextRow.get('class') == ['tablesorter-childRow']:
                                cells = nextRow.findAll('td')
                                tmp_data.update({
                                    'response': cells[1].get_text().strip(),
                                    'confidence': cells[2].get_text().strip(),
                                    'comment': cells[3].get_text().strip(),
                                })
                        
                        data += [tmp_data]
                
        col_order = ['survey_date', 'topic_name', 'topic_url', 'survey_question', 'economist_name', 'economist_url', 'economist_headshot', 'institution', 'response', 'confidence', 'comment']
        df = pd.DataFrame(data, columns=col_order) \
               .assign(
                   confidence = lambda x: x['confidence'] \
                                .replace(r'[^\d]+|^$', np.nan, regex=True).astype(float),
                )
        df.to_pickle(os.path.join(os.path.dirname(__file__), 'igmchicago.pkl')) 
    else:
        df = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'igmchicago.pkl'))

    ###########
    # STORAGE #
    ###########
        
    import sqlalchemy
    engine = sqlalchemy.create_engine('sqlite:///igmchicago.db')
    df.to_sql('igmchicago', engine, index=False, if_exists='replace')
    if os.path.exists('igmchicago.db'):
        df = pd.read_sql_table('igmchicago', engine)
        
    cnxn = engine.connect()
    r = cnxn.execute("""
		WITH igm AS (
			SELECT economist_name, institution, AVG(confidence) AS avg_conf
			FROM igmchicago
			GROUP BY economist_name, institution
		)
		SELECT * FROM igm i
		WHERE avg_conf >
			(SELECT AVG(avg_conf) FROM igm g WHERE i.institution = g.institution)
		ORDER BY institution, avg_conf
    """)
    r.fetchall()

    #######
    # EDA #
    #######
        
    df.columns
    df.shape
    df.describe()
    df.info()
    any(df.duplicated())
    df.head(10)
    df.tail(2)
    df['institution'].value_counts().plot(kind='bar')

    ##############
    # VALIDATION #
    ##############
    
    # Correct response types
    df.loc[df['response'].str.contains(r'did not', case=False) | df['response'].str.contains(r'---'), 'response'] = np.nan
    
    # Convert empty string comments into null types
    df['comment'] = df['comment'].replace(r'^$', np.nan, regex=True)

    # Assign sex variable to economists
    sex = pd.read_csv(os.path.join(os.path.dirname(__file__), 'economist_sex_mapping.csv'), index_col='economist_name')
    df['sex'] = df['economist_name'].map(sex['sex'])

    # Assign response categories to numerical values    
    certainty_mapping = {
        'Strongly Disagree': -2, 
        'Disagree': -1,
        'Uncertain': 0,
        'No opinion': 0,
        'Agree': 1,
        'Strongly Agree': 2,
    }
    df = df.assign(response_int = lambda x: x['response'].map(certainty_mapping))

    ############
    # ANALYSIS #
    ############
    
    facet_labels = ['economist_name', 'institution', 'sex']
    facets = df.groupby(facet_labels).first().reset_index()[facet_labels]
    
    # Summary statistics
    len(df.groupby('topic_name')) # 132 topics
    len(df.groupby(['topic_name', 'survey_question'])) # 195 survey questions
    len(facets['economist_name'].unique()) # 51 economists
    facets.groupby('sex').size() # 11 female, 40 male
    facets.groupby('institution').size().sort_values(ascending=False)
    
    #################
    # VISUALIZATION #
    #################
    
    def colored_bar_plot(df, y, color_map, xlab, ylab, title, ylim=(0,100), legend_loc='upper left', skip_bars=0, swap_bg=False, rotate=False, ticks=False):            
        ax = df.plot(kind='bar', x='economist_name', y=y, legend=False)
        ax.set(xlabel=xlab, ylabel=ylab, ylim=ylim)
        plt.title(title)
        
        # Extend bottom of plot
        plt.gcf().subplots_adjust(bottom=0.35)
           
        # Set bar colors
        barlist = [x for x in ax.get_children() if isinstance(x, matplotlib.patches.Rectangle)]
        for i, bar in enumerate(barlist[skip_bars:-1]):
            bar.set_color(df.iloc[i+skip_bars]['colors'])
            
        # Add a custom legend
        patches = []
        for group, color in color_map.items():
            patches += [mpatches.Patch(color=color, label=group)]
        ax.legend(handles=patches, loc=legend_loc, shadow=True)
        
        # Lighten background color
        if swap_bg:
            light_grey = np.array([250, 250, 245]) / 255.
            barlist[-1].set_color(light_grey)
        
        # Rotate x-axis labels
        if rotate:
            ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=37.5, ha='right')
            ax.tick_params(axis='x', pad=5)
        
        # Add tick marks to bottom
        if ticks:
            for line in ax.xaxis.get_ticklines():
                line.set_color('grey')
                line.set_markersize(5)    
            ax.xaxis.tick_bottom()
            
        return ax
                
    fancy_colored_bar_plot = partial(colored_bar_plot, swap_bg=True, rotate=True, ticks=True)
                
    def get_color_scheme(series, colors=None):
        groups = series.dropna().unique()
        groups.sort()
        if colors is None:
            start, spacer = 50, 50
            colors = matplotlib.cm.Blues(range(start + len(groups)*spacer), bytes=True)[start::spacer] / 255.
            
        color_map = OrderedDict((k, v) for k, v in zip(groups, colors[:len(groups)]))
        # color_map = { k:v for k, v in zip(groups, colors[:len(groups)]) }
        return pd.Series( series.map(color_map) ), color_map
        
    def scatter_plot(df, x, y, xlab, ylab):
        ax = df.plot(kind='scatter', x=x, y=y)
        for k, v in df.iterrows():
            ax.annotate(k, v, xytext=(5,2), textcoords='offset points',  fontsize=10)
        ax.set(xlabel=xlab, ylabel=ylab)
        return ax
        
    #/# Response distribution by economist_name/institution #/#
    colors = np.array([(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]) / 255.
    facets['colors'], color_map = get_color_scheme(facets['institution'], colors)
    response_dist = df.groupby('economist_name').count().assign(
            pct = lambda df: (df['response'] / df['topic_name']) * 100
        ) \
        .sort_values(by='pct') \
        .take([-1], axis=1) \
        .merge(facets, left_index=True, right_on='economist_name')  
    fancy_colored_bar_plot(response_dist, 'pct', color_map, 'economist', 'response rate (%)', 'Proportion of surveys answered by economist', ylim=(40,100))

    #/# Comment length (when they do comment) by gender/institution/economist_name #/#
    commenters = df.groupby('economist_name').count()['comment'].sort_values()
    length = df.groupby('economist_name').apply(lambda x: np.mean(x['comment'].str.len())).sort_values()
    comments = pd.concat([commenters, length], axis=1).sort_values('comment')
    comments.columns = ['num_comments', 'avg_comment_length_chars']
    comments.index.name = 'economist_name'
    comments['bins'] = pd.qcut(comments['avg_comment_length_chars'], 5)    
    comments['colors'], color_map = get_color_scheme(comments['bins'])
    fancy_colored_bar_plot(comments.reset_index(), 'num_comments', color_map, 'economist', 'number of comments', 'Number of comments by economist (color = average comment length)', skip_bars=1) 

    #/# Confidence vs. volatility #/#
    conf_stats = df[['economist_name', 'confidence']].groupby('economist_name').agg([np.mean, np.std]).sort_values([('confidence', 'mean')]) # sort MultiIndex via tuple
    conf_stats.columns = conf_stats.columns.droplevel(0)
    scatter_plot(conf_stats, x='mean', y='std', xlab='mean confidence in response', ylab='standard deviation of confidence')
    
    #/# Saltwater vs. freshwater economics (last 30 surveys) #/#    
    schools = df[df['institution'].str.contains('Harvard|Chicago')] \
        .groupby(['survey_date', 'topic_name', 'survey_question', 'institution']) \
        .mean()['response_int'] \
        .unstack() \
        .tail(30) \
        .sort_index(ascending=False)
    schools.index = schools.index.droplevel('survey_date') # survey_date needed to order DataFrame
    ax = schools.plot(kind='barh', xlim=(-2,2))
    ax.set(ylabel='topic name, survey question')
    ax.set_xticklabels(['Strongly disagree', '', 'Disagree', '', 'Uncertain/No opinion', '', 'Agree', '', 'Strongly Agree'])
    plt.gcf().subplots_adjust(left=0.2)
    
    assert False
    
    """ Visualizations to skip """
    
    #/# Certainty vs. uncertainty by economist #/# (skip)
    response_proportions = df.groupby(['economist_name', 'response']).size() / df.groupby('economist_name').size() 
    uncertainty = response_proportions.loc[pd.IndexSlice[:, 'Uncertain']].rename('uncertainty')
    certainty = response_proportions.loc[pd.IndexSlice[:, ['Strongly Agree', 'Strongly Disagree']]].groupby(level=0).sum().rename('certainty')
    s = pd.DataFrame([certainty, uncertainty]).T.sort_values('uncertainty') * 100        
    scatter_plot(s, x='certainty', y='uncertainty', xlab='strongly agree or disagree (%)', ylab='uncertain (%)')
    ax.set(ylabel='confidence in response (mean)')
    
    #/# Sample of economists: frequency of each response type #/# (skip)
    response_proportions = response_proportions.rename('freq')
    mask = response_proportions.loc[pd.IndexSlice[:, 'Uncertain']] > .2
    subset = response_proportions.loc[tuple(mask[mask].index), :] * 100
    ax = sns.barplot(x='economist_name', y='freq', hue='response', data=subset.reset_index())
    plt.gcf().subplots_adjust(bottom=0.35)
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
    ax.set(xlabel='economist', ylabel='response frequency (%)')
    ax.legend(title=None)
    
    #/# Voting for or against the average (ie. contrarian) #/# (skip)
    economist_confidence = df.groupby(['economist_name', 'topic_name', 'survey_question']).mean().drop('confidence', axis=1).rename(columns={'response': 'mean_survey_response'}).unstack(0)
    topic_mean_confidence = df.groupby(['topic_name', 'survey_question']).mean().drop('confidence', axis=1).rename(columns={'response': 'mean_survey_response'})
    abs_diffs = np.abs(economist_confidence.subtract(topic_mean_confidence, axis='columns', level=0))
    contrarian_name = abs_diffs.mean().idxmax()
    contrarian = df.loc[df['economist_name'] == contrarian_name[1]].groupby(['topic_name', 'survey_question']).mean().drop('confidence', axis=1)
    plot = topic_mean_confidence.join(contrarian).rename(columns={'response': 'contrarian'})
    ax = plot.plot(style='o')
    ax.margins(.05)
    
    #/# Caroline review #/# (skip)
    c = df[df['economist_name'].str.contains('Caroline')]
    ax = c.groupby('survey_date').mean().plot(style='ko', legend=None)
    ax.margins(.05) # 5% padding in all directions
    ax.set_xlim(pd.Timestamp('2011-09-01'), pd.Timestamp('2016-07-01'))
    ax.set(ylabel='confidence', xlabel='survey date')    
    
    #/# How often does high confidence in uncertainty appear? #/# (skip)
    t = df.groupby('response').mean()['confidence'].sort_values(na_position='first') # the trend we expect
    ax = sns.barplot(x='response', y='confidence', data=t.reset_index())

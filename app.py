import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html, Input, Output, State
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import numpy as np

# Load the DataFrame from the .pkl file
pkl_file = "deviations_dataframe_extended.pkl"
df = pd.read_pickle(pkl_file)

# Select columns that contain '_embedding_cluster' and do not contain 'date'
allowed_cols_init = ['title', 'type', 'level', 'location', 'observer_or_identifier', 
                'how_did_deviation_occur', 'immediate_action_taken', 'investigation_additional_details',
                'product_identification','root_cause', 'root_cause_category', 'product_impact_assessement']
allowed_cols = [col + "_embedding_cluster" for col in allowed_cols_init]
filtered_df = df[allowed_cols]
corr_matrix = filtered_df.corr()

# UTIL FUNCTIONS
client = OpenAI(api_key='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')

# OpenAI Embeddings
def get_openai_embeddings(texts):
    response = client.embeddings.create(input=texts, model="text-embedding-ada-002")
    embeddings = [data.embedding for data in response.data]
    return embeddings

def search_over_embeddings(df, search_query, top_x=5):
    # Get the search query embedding
    query_embedding = get_openai_embeddings([search_query])[0]
    
    # Initialize a list to store similarity scores
    similarity_scores = []
    
    # Iterate through each embedding column in the dataframe
    for column in df.columns:
        if column.endswith('_embedding'):
            # Get the embeddings for the current column
            column_embeddings = np.vstack(df[column].values)
            
            # Calculate cosine similarity between query embedding and column embeddings
            similarities = cosine_similarity([query_embedding], column_embeddings)[0]
            
            # Store the similarity scores along with the column name
            similarity_scores.append((similarities, column))
    
    # Combine similarity scores into a single array
    combined_similarities = np.hstack([score for score, _ in similarity_scores])
    
    # Get the top x indices with the highest similarity scores
    top_indices = np.argsort(combined_similarities)[-top_x:][::-1]
    
    # Get the rows and column names with the highest similarity scores
    results = []
    for idx in top_indices:
        row_idx = idx % len(df)
        col_idx = idx // len(df)
        similarity_score = combined_similarities[idx]
        column_name = similarity_scores[col_idx][1]
        
        results.append((df.iloc[row_idx][allowed_cols_init], column_name, similarity_score))
    
    return results

# Function to summarize cluster values using GPT-3.5-turbo
def answer_with_context(query, context):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are analyzing trends in deviations from a manufacturing process. Answer the user's \
                                          question using the provided context in a thoughtful and analytical manner."},
            {"role": "user", "content": f"Here is some context on previous, related deviations: {context}. \n The user's question is {query}"}
        ],
    )
    answer = completion.choices[0].message.content.strip()
    return answer

# basic answer
def answer_with_prompt(prompt):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful and thoughtful assistant. Assist the user with their query."},
            {"role": "user", "content": prompt}
        ],
    )
    answer = completion.choices[0].message.content.strip()
    return answer

# APP CODE STARTS HERE 

# Create the correlation heatmap
heatmap = go.Heatmap(
    z=corr_matrix.values,
    x=[i.replace("_embedding_cluster", "") for i in corr_matrix.columns],
    y=[i.replace("_embedding_cluster", "") for i in corr_matrix.columns],
    colorscale='Viridis',
    text=corr_matrix.values,
    hoverinfo='text'
)

heatmap_fig = go.Figure(data=[heatmap])
heatmap_fig.update_layout(
    title_x=0.5,
    width=1200,
    height=1000,
    xaxis_nticks=len(df.columns),
    yaxis_nticks=len(df.columns),
    margin=dict(l=200, r=200, t=100, b=200),
)
heatmap_fig.update_xaxes(tickangle=45, tickfont=dict(size=12))
heatmap_fig.update_yaxes(tickfont=dict(size=12))

# Dash application
app = Dash(__name__)

app.layout = html.Div(
    style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'padding': '20px'},
    children=[
        html.Div(
            style={'display': 'flex', 'alignItems': 'center'},
            children=[
                dcc.Input(
                    id='search-input', 
                    type='text', 
                    placeholder='Enter search text',
                    style={'width': '800px', 'padding': '10px', 'marginRight': '10px'}
                ),
                html.Button(
                    'Search', 
                    id='search-button', 
                    n_clicks=0,
                    style={'padding': '10px'}
                ),
                # html.Button(
                #     'Clear', 
                #     id='clear-button', 
                #     n_clicks=0,
                #     style={'padding': '10px', 'marginLeft': '10px', 'display': 'none'}
                # ),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        html.A('📁', style={'fontSize': '24px'})  # Small upload icon
                    ]),
                    style={
                        'width': '30px',
                        'height': '30px',
                        'lineHeight': '30px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    multiple=False
             )
            ]
        ),
         dcc.Textarea(
            id='ai-response', 
            style={'width': '70%', 'height': '200px', 'display': 'block'},
            readOnly=True
        ),
        dcc.Graph(id='heatmap', figure=heatmap_fig),
        html.Div(id='graphs-container', children=[
            dcc.Graph(id='scatter-plot', style={'height': '800px', 'width': '80%', 'margin': '20px', 'display': 'none'}),
            dcc.Graph(id='histogram', style={'height': '800px', 'width': '80%', 'margin': '20px', 'display': 'none'}),
            #html.Button('Delete Plot', id='delete-plot-button', style={'margin': '20px', 'display': 'none'})
        ]),
        html.Div(
            id='summary-output', 
            style={
                'whiteSpace': 'pre-line', 
                'fontSize': '18px',
                'padding': '20px', 
                'borderRadius': '10px',
                'width': '80%'
            }
        ),
    ]
)

@app.callback(
    [Output('scatter-plot', 'figure'), Output('scatter-plot', 'style'), 
     Output('histogram', 'figure'), Output('histogram', 'style'),
     #Output('delete-plot-button', 'style')
     ], 
    [Input('heatmap', 'clickData'), 
     #Input('delete-plot-button', 'n_clicks')
     ],
    [State('scatter-plot', 'clickData'), State('heatmap', 'clickData')]
)
def display_plot(heatmap_clickData, scatter_clickData, heatmap_clickData_state):
    ctx = dash.callback_context
    if not ctx.triggered:
        return {}, {'display': 'none'}, {}, {'display': 'none'}
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # if triggered_id == 'delete-plot-button':
    #     return {}, {'display': 'none'}, {}, {'display': 'none'}, {'display': 'none'}
    
    if triggered_id == 'heatmap' and heatmap_clickData:
        x_feature = heatmap_clickData['points'][0]['x'] + "_embedding_cluster"
        y_feature = heatmap_clickData['points'][0]['y'] + "_embedding_cluster"
        if x_feature == y_feature:
            feature = x_feature.replace('_embedding_cluster', '')
            df['date_of_ocurrence'] = pd.to_datetime(df['date_of_ocurrence'])
            df['month'] = df['date_of_ocurrence'].dt.to_period('M').astype(str)
            monthly_df = df.groupby(['month', feature + '_embedding_cluster']).size().reset_index(name='count')
            histogram_fig = px.histogram(
                monthly_df, x='month', y='count', color=feature + "_embedding_cluster", barmode='stack',
                title='',#f'Stacked Histogram of {feature} over Time', 
                width=1200, height=800
            )
            #histogram_fig.add_traces(px.line(monthly_df, x='month', y='count', color=feature + "_embedding_cluster").data)
            histogram_fig.update_layout(clickmode='event+select')
            return {}, {'display': 'none'}, histogram_fig.to_dict(), {'display': 'block'}
        else:
            scatter_fig = px.scatter(
                df, x=x_feature, y=y_feature,
                title='', width=1200, height=800
            )
            scatter_fig.update_traces(marker=dict(size=12), selector=dict(mode='markers'))
            scatter_fig.update_layout(clickmode='event+select')
            return scatter_fig.to_dict(), {'display': 'block'}, {}, {'display': 'none'}
    return {}, {'display': 'none'}, {}, {'display': 'none'}

@app.callback(
    [Output('summary-output', 'children'), Output('scatter-plot', 'clickData'), Output('histogram', 'clickData')],
    [Input('scatter-plot', 'clickData'), Input('histogram', 'clickData')],
    [State('heatmap', 'clickData')]
)
def display_summary(scatter_click_data, histogram_click_data, heatmap_click_data):
    summary_text = ""
    if scatter_click_data and heatmap_click_data:
        point_index = scatter_click_data['points'][0]['pointIndex']
        x_feature = heatmap_click_data['points'][0]['x']
        y_feature = heatmap_click_data['points'][0]['y']
        
        x_summary_col = x_feature.replace('_embedding_cluster', '_summary')
        y_summary_col = y_feature.replace('_embedding_cluster', '_summary')
        
        prompt = "Characterize a subset (a cluster) of the following feature: {}. \n This is a list of descriptions within this feature belonging to the same cluster: {}. \n summarize the descriptions and remove redunancies \
                in 1 sentence or less to provide a cohesive and clear summary of this feature's and this cluster's attributes."
        x_summary = answer_with_prompt(prompt.format(x_feature, df.loc[point_index, x_summary_col]))
        y_summary = answer_with_prompt(prompt.format(y_feature, df.loc[point_index, y_summary_col]))
        
        summary_text = f"Summary for {x_feature} (Cluster {x_feature}):\n{x_summary}\n\nSummary for {y_feature} (Cluster {y_feature}):\n{y_summary}"
    
    if histogram_click_data and heatmap_click_data:

        point_data = histogram_click_data['points'][0]
        cluster_value = point_data['curveNumber']  # This is the index of the curve in the plotly figure data
        feature = heatmap_click_data['points'][0]['x']
        summary_col = feature + '_summary'
        filtered_df = df[df[feature + '_embedding_cluster'] == cluster_value]
        summary_texts = filtered_df[summary_col].tolist()
        prompt = "Remove redunant statements or information from this and return the result in 1 sentence or less: {}"
        summary_texts = answer_with_prompt(prompt.format(summary_texts))
        
        summary_text = f"Summary for {feature} (Cluster {cluster_value}):\n" + summary_texts
    
        
    return summary_text, None, None

@app.callback(
    [Output('ai-response', 'value'), Output('ai-response', 'style')],
    [Input('search-button', 'n_clicks')],
    [State('search-input', 'value')]
)
def handle_search_and_clear(search_clicks, query):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "", {'display': 'none'}
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # if triggered_id == 'clear-button':
    #     return "", {'display': 'none'}, {'display': 'none'}

    if triggered_id == 'search-button' and query:
        results = search_over_embeddings(df, query, top_x=10)
        response = answer_with_context(query, [res[0].to_string() for res in results])
        return response, {'display': 'block'}
    
    return "", {'display': 'none'}

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)

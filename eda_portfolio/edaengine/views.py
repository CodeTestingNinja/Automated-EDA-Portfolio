from django.shortcuts import render, redirect
from django.http import HttpResponse
import pandas as pd
import base64
import io # <-- Import the io library
import sys # <-- Import sys to capture df.info() output
# import matplotlib.pyplot as plt
# import seaborn as sns
from django.template.loader import get_template
from xhtml2pdf import pisa


# Helper function
def update_session_with_df(request, df):
    """
    Takes a DataFrame, calculates all necessary stats,
    and updates the user's session.
    """
    print('--- Updating session with new DataFrame ---')

    # request.session['previous_csv_base64'] = request.session.get('csv_base64', None)

    new_csv_bytes = df.to_csv(index=False).encode('utf-8')
    new_csv_b64 = base64.b64encode(new_csv_bytes).decode('utf-8')
    request.session['csv_base64'] = new_csv_b64

    request.session['table'] = df.head().to_html(classes="table table-bordered")
    request.session['shape'] = df.shape
    request.session['descriptive_stats'] = df.describe().to_html(classes="table table-bordered")

    missing_values_df = df.isnull().sum().to_frame('missing_count')
    request.session['missing_values'] = missing_values_df.to_html(classes="table table-bordered")

    buffer = io.StringIO()
    df.info(buf=buffer)
    request.session['info'] = buffer.getvalue()

    print("--- Session update complete ---")


def action_undo(request):
    """
    Restores the DataFrame from the 'previous_csv_base64' session key.
    """
    previous_b64 = request.session.get('previous_csv_base64')

    if previous_b64:
        request.session['csv_base64'] = previous_b64

        csv_bytes = base64.b64decode(previous_b64)
        df = pd.read_csv(io.BytesIO(csv_bytes))
        update_session_with_df(request, df)

        request.session['previous_csv_base64'] = None
    
    return redirect(request.META.get('HTTP_REFERER', 'analysis_home'))


def action_download_csv(request):
    """
    Retrieves the current DataFrame from the session and returns it as a downloadable CSV file.
    """
    csv_b64 = request.session.get('csv_base64')
    if not csv_b64:
        return HttpResponse("No data available to download.", status=404)
    
    # Decode and get the CSV data as a string
    csv_bytes = base64.b64decode(csv_b64)
    csv_string = csv_bytes.decode('utf-8')

    # Create the HTTP response with the correct content type for CSV
    response = HttpResponse(csv_string, content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="cleaned_dataset.csv"'
    
    return response


# Home Page view
def home(request):
    return render(request, "edaengine/home.html")



# Upload Page view
def upload_file(request):
    if request.method == "POST":
        file = request.FILES.get('file')
        if not file:
            return render(request, "edaengine/upload.html", {
                "message": "❌ No file uploaded. Please choose a file."
            }) 

        try:
            file_bytes = file.read() # Read the file content ONCE into a bytes variable
            df = pd.read_csv(io.BytesIO(file_bytes))
            table = df.head().to_html(classes="table table-bordered")
            shape = df.shape
            descriptive_stats = df.describe().to_html(classes="table table-bordered")
            missing_values_df = df.isnull().sum().to_frame('missing_count')
            missing_values = missing_values_df.to_html(classes="table table-bordered")

            # Get info by capturing the print output
            buffer = io.StringIO()
            df.info(buf=buffer)
            info = buffer.getvalue()

            # Encode the original file bytes for storage
            csv_b64 = base64.b64encode(file_bytes).decode('utf-8')

            # Store everything in the session
            request.session['csv_base64'] = csv_b64
            request.session['table'] = table
            request.session['shape'] = shape
            request.session['info'] = info
            request.session['descriptive_stats'] = descriptive_stats
            request.session['missing_values'] = missing_values

            # Pass the preview of the table to the upload page
            return redirect('analysis_tool', tool_name='analysis_overview')
        
        except Exception as e:
            return render(request, "edaengine/upload.html", {
                "message": f"❌ Error reading file: {e}"
            })
    
    return render(request, "edaengine/upload.html")



# Analysis Workspace Page
def analysis(request, tool_name=None):
    if not request.session.get('csv_b64'):
        return HttpResponse("⚠️ No dataset uploaded. Please upload first.")
    
    # The list of tools for the sidebar
    tools = [
        {'name': 'Initial Overview', 'url_name': 'analysis_overview', 'icon': 'fas fa-search'},
        {'name': 'Missing Values', 'url_name': 'analysis_missing', 'icon': 'fas fa-exclamation-triangle'},
        {
            'name': 'Visualizations',
            'icon': 'fas fa-chart-bar',
            'sub_items': [
                {
                    'name': 'Univariate',
                    'sub_items': [
                        {'name': 'Histogram', 'url_name': 'viz_histogram'},
                        {'name': 'Count Plot', 'url_name': 'viz_countplot'},
                        {'name': 'Box Plot', 'url_name': 'viz_boxplot'},
                        {'name': 'KDE Plot', 'url_name': 'viz_kdeplot'},
                    ]
                },
                {
                    'name': 'Bivariate',
                    'sub_items': [
                        {'name': 'Scatter Plot', 'url_name': 'viz_scatterplot'},
                        {'name': 'Bar Plot', 'url_name': 'viz_barplot'},
                        {'name': 'Line Plot', 'url_name': 'viz_lineplot'},
                    ]
                },
                {
                    'name': 'Multivariate',
                    'sub_items': [
                        {'name': 'Correlation Heatmap', 'url_name': 'viz_heatmap'},
                        {'name': 'Pair Plot', 'url_name': 'viz_pairplot'},
                    ]
                }
            ]
        },
        {'name': 'Outlier Handling', 'url_name': 'analysis_outliers', 'icon': 'fas fa-cut'},
        {'name': 'Summary & Report', 'url_name': 'analysis_summary', 'icon': 'fas fa-file-download'},
        # Add more top-level tools here later
    ]

    flash_message = request.session.pop('flash_message', None)

    context = {
        'tools': tools,
        'selected_tool': tool_name,
        'partial_template_name': None,
        'flash_message': flash_message,
    }

    if tool_name == 'analysis_overview':
        context['partial_template_name'] = 'edaengine/analysis_partials/_initial_overview.html'
        # Get the data ready for partial
        context['table'] = request.session.get('table')
        context['shape'] = request.session.get('shape')
        context['info'] = request.session.get('info')
        context['descriptive_stats'] = request.session.get('descriptive_stats')
        context['missing_values'] = request.session.get('missing_values')


    elif tool_name == 'analysis_missing':
        context['partial_template_name'] = 'edaengine/analysis_partials/_missing_values.html'
        
        csv_b64 = request.session.get('csv_base64')
        csv_bytes = base64.b64decode(csv_b64)
        df = pd.read_csv(io.BytesIO(csv_bytes))

        missing_data = df.isnull().sum()
        columns_with_missing = missing_data[missing_data > 0]

        missing_info_list = []
        for col_name, missing_count in columns_with_missing.items():
            missing_info_list.append({
                'name': col_name,
                'missing_count': missing_count,
                'dtype': df[col_name].dtype,
                'is_numeric': pd.api.types.is_numeric_dtype(df[col_name]),
            })
        context['missing_info'] = missing_info_list

        # This part handles the form submission
        if request.method == 'POST':
            request.session['previous_csv_base64'] = request.session.get('csv_base64')
            
            columns_to_drop = []

            for col_info in missing_info_list:
                col_name = col_info['name']
                action = request.POST.get(f'action_{col_name}')

                # Action Block
                if action == 'drop_rows':
                    df.dropna(subset=[col_name], inplace=True)

                elif action == 'drop_column':
                    # Don't drop it yet, just add its name to our list
                    columns_to_drop.append(col_name)

                elif action == 'fill_mean':
                    mean_val = df[col_name].mean()
                    df[col_name].fillna(mean_val, inplace=True)
            
                elif action == 'fill_median':
                    median_val = df[col_name].median()
                    df[col_name].fillna(median_val, inplace=True)
            
                elif action == 'fill_mode':
                    mode_val = df[col_name].mode()[0]
                    df[col_name].fillna(mode_val, inplace=True)

            if columns_to_drop:
                df.drop(columns=columns_to_drop, inplace=True)
                print(f"Dropped columns: {columns_to_drop}")

            update_session_with_df(request, df)
            request.session['flash_message'] = "Missing value actions applied successfully."

            # Redirect back to the same page
            return redirect('analysis_tool', tool_name='analysis_missing')
        
        # context['missing_values_table'] = request.session.get('missing_values')


    elif tool_name == 'viz_histogram':
        import matplotlib
        matplotlib.use('Agg') # Use the non-interactive 'Agg' backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        context['partial_template_name'] = 'edaengine/analysis_partials/_viz_histogram.html'

        csv_b64 = request.session.get('csv_base64')
        csv_bytes = base64.b64decode(csv_b64)
        df = pd.read_csv(io.BytesIO(csv_bytes))

        numerical_cols = df.select_dtypes(include=['number']).columns

        analysis_cards = []

        for col in numerical_cols:
            card_data = {}
            col_data = df[col]

            stats = {
                'Count': col_data.count(),
                'Missing': col_data.isnull().sum(),
                'Mean': f"{col_data.mean():.2f}",
                'Std Dev': f"{col_data.std():.2f}",
                'Min': col_data.min(),
                '25%': col_data.quantile(0.25),
                '50% (Median)': col_data.median(),
                '75%': col_data.quantile(0.75),
                'Max': col_data.max(),
            }

            card_data['stats_html'] = pd.Series(stats).to_frame('Value').to_html(classes="table table-sm table-bordered")

            plt.figure(figsize=(6, 4))
            sns.histplot(col_data, kde=True, bins=30)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel('Frequency')

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)

            card_data['plot_b64'] = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()

            card_data['column_name'] = col
            analysis_cards.append(card_data)

        context['analysis_cards'] = analysis_cards
    

    elif tool_name == 'viz_countplot':
        import matplotlib
        matplotlib.use('Agg') # Use the non-interactive 'Agg' backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        context['partial_template_name'] = 'edaengine/analysis_partials/_viz_countplot.html'

        csv_b64 = request.session.get('csv_base64')
        csv_bytes = base64.b64decode(csv_b64)
        df = pd.read_csv(io.BytesIO(csv_bytes))

        categorical_cols = df.select_dtypes(include=['object']).columns

        analysis_cards = []
        CARDINALITY_THRESHOLD = 50

        for col in categorical_cols:
            card_data = {}
            col_data = df[col]

            stats = {
                'Count': col_data.count(),
                'Unique': col_data.nunique(),
                'Missing': col_data.isnull().sum(),
                'Top (Mode)': col_data.mode()[0] if not col_data.empty else 'N/A',
                'Freq of Top': col_data.value_counts().iloc[0] if not col_data.empty else 'N/A',
            }
            card_data['stats_html'] = pd.Series(stats).to_frame('Value').to_html(classes="table table-sm table-bordered")

            if col_data.nunique() < CARDINALITY_THRESHOLD:
                print(f"Generating stats & plots for {col} (Cardinality: {df[col].nunique()})")
                
                plt.figure(figsize=(10, 6))
                sns.countplot(y=col, data=df, order=df[col].value_counts().index)
                plt.title(f'Count of {col}')
                plt.xlabel('Count')
                plt.ylabel(col)
                plt.tight_layout()
                    
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                
                card_data['plot_b64'] = base64.b64encode(buffer.read()).decode('utf-8')
                plt.close()

            else:
                # For high cardinality, we won't generate a plot, so we can set it to None
                print(f"Skipping plot for {col} due to high cardinality")
                card_data['plot_b64'] = None

            card_data['column_name'] = col
            analysis_cards.append(card_data)

        context['analysis_cards'] = analysis_cards
    

    elif tool_name == 'viz_boxplot':
        import matplotlib
        matplotlib.use('Agg') # Use the non-interactive 'Agg' backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        context['partial_template_name'] = 'edaengine/analysis_partials/_viz_boxplot.html'
        
        csv_b64 = request.session.get('csv_base64')
        csv_bytes = base64.b64decode(csv_b64)
        df = pd.read_csv(io.BytesIO(csv_bytes))

        numerical_cols = df.select_dtypes(include=['number']).columns

        analysis_cards = []
        for col in numerical_cols:
            card_data = {}
            col_data = df[col]

            stats = {
                'Count': col_data.count(), 
                'Missing': col_data.isnull().sum(),
                'Mean': f"{col_data.mean():.2f}", 
                'Std Dev': f"{col_data.std():.2f}",
                'Min': col_data.min(), 
                '25%': col_data.quantile(0.25),
                '50% (Median)': col_data.median(), 
                '75%': col_data.quantile(0.75),
                'Max': col_data.max(),
            }
            card_data['stats_html'] = pd.Series(stats).to_frame('Value').to_html(classes="table table-sm table-bordered")

            plt.figure(figsize=(8, 4))
            sns.boxplot(x=col_data)
            plt.title(f"Box Plot of {col}")

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            card_data['plot_b64'] = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            card_data['column_name'] = col
            analysis_cards.append(card_data)
            
        context['analysis_cards'] = analysis_cards


    elif tool_name == 'viz_kdeplot':
        import matplotlib
        matplotlib.use('Agg') # Use the non-interactive 'Agg' backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        context['partial_template_name'] = 'edaengine/analysis_partials/_viz_kdeplot.html'
        
        csv_b64 = request.session.get('csv_base64')
        csv_bytes = base64.b64decode(csv_b64)
        df = pd.read_csv(io.BytesIO(csv_bytes))

        numerical_cols = df.select_dtypes(include=['number']).columns
        
        analysis_cards = [] 
        for col in numerical_cols:
            card_data = {}
            col_data = df[col]

            # Re-using the same stats block
            stats = {
                'Count': col_data.count(), 'Missing': col_data.isnull().sum(),
                'Mean': f"{col_data.mean():.2f}", 'Std Dev': f"{col_data.std():.2f}",
                'Min': col_data.min(), '25%': col_data.quantile(0.25),
                '50% (Median)': col_data.median(), '75%': col_data.quantile(0.75),
                'Max': col_data.max(),
            }
            card_data['stats_html'] = pd.Series(stats).to_frame('Value').to_html(classes="table table-sm table-bordered")

            # Generate KDE Plot
            plt.figure(figsize=(8, 4))
            sns.kdeplot(data=col_data, fill=True) # `fill=True` shades the area under the curve
            plt.title(f'Density of {col}')
            plt.xlabel(col)
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            card_data['plot_b64'] = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            card_data['column_name'] = col
            analysis_cards.append(card_data)
            
        context['analysis_cards'] = analysis_cards


    elif tool_name == 'viz_scatterplot':
        import matplotlib
        matplotlib.use('Agg') # Use the non-interactive 'Agg' backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        context['partial_template_name'] = 'edaengine/analysis_partials/_viz_scatterplot.html'

        csv_b64 = request.session.get('csv_base64')
        csv_bytes = base64.b64decode(csv_b64)
        df = pd.read_csv(io.BytesIO(csv_bytes))

        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        context['numerical_columns'] = numerical_columns

        x_axis = request.GET.get('x_axis')
        y_axis = request.GET.get('y_axis')

        if x_axis and y_axis:
            plt.figure(figsize=(10, 7))
            sns.regplot(x=df[x_axis], y=df[y_axis], line_kws={"color": "red"})
            plt.title(f'Scatter Plot of {y_axis} vs. {x_axis}')
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            image_b64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            context['plot_b64'] = image_b64
    

    elif tool_name == 'viz_barplot':
        import matplotlib
        matplotlib.use('Agg') # Use the non-interactive 'Agg' backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        context['partial_template_name'] = 'edaengine/analysis_partials/_viz_barplot.html'

        csv_b64 = request.session.get('csv_base64')
        csv_bytes = base64.b64decode(csv_b64)
        df = pd.read_csv(io.BytesIO(csv_bytes))

        # Getting both the columns for the dropdowns
        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = [col for col in df.select_dtypes(include=['object']).columns if df[col].nunique() < 50]

        context['numerical_columns'] = numerical_columns
        context['categorical_columns'] = categorical_columns

        cat_col = request.GET.get('cat_col')
        num_col = request.GET.get('num_col')

        context['selected_cat'] = cat_col
        context['selected_num'] = num_col

        if cat_col and num_col:
            plt.figure(figsize=(10, 5))
            sns.barplot(x=cat_col, y=num_col, data=df)

            plt.title(f'Average {num_col} by {cat_col}')
            plt.xlabel(cat_col)
            plt.ylabel(f'Average {num_col}')
            plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability
            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
        
            image_b64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
        
            context['plot_b64'] = image_b64


    elif tool_name == 'viz_lineplot':
        import matplotlib
        matplotlib.use('Agg') # Use the non-interactive 'Agg' backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        context['partial_template_name'] = 'edaengine/analysis_partials/_viz_lineplot.html'
        
        csv_b64 = request.session.get('csv_base64')
        csv_bytes = base64.b64decode(csv_b64)
        df = pd.read_csv(io.BytesIO(csv_bytes))

        context['all_columns'] = df.columns.tolist()
        context['numerical_columns'] = df.select_dtypes(include=['number']).columns.tolist()

        date_col = request.GET.get('date_col')
        num_col = request.GET.get('num_col')

        context['selected_date'] = date_col
        context['selected_num'] = num_col

        if date_col and num_col:
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                df_sorted = df.sort_values(by=date_col)

                plt.figure(figsize=(14, 7))
                sns.lineplot(x=date_col, y=num_col, data=df_sorted)
                plt.title(f'{num_col} over Time ({date_col})')
                plt.xlabel(date_col)
                plt.ylabel(num_col)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                
                image_b64 = base64.b64encode(buffer.read()).decode('utf-8')
                plt.close()

                context['plot_b64'] = image_b64
            
            except Exception as e:
                context['error_message'] = f"Could not convert column '{date_col}' to a date format. Please ensure it contains valid dates. Error: {e}"


    elif tool_name == 'viz_heatmap':
        import matplotlib
        matplotlib.use('Agg') # Use the non-interactive 'Agg' backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        context['partial_template_name'] = 'edaengine/analysis_partials/_viz_heatmap.html'

        csv_b64 = request.session.get('csv_base64')
        csv_bytes = base64.b64decode(csv_b64)
        df = pd.read_csv(io.BytesIO(csv_bytes))

        corr = df.select_dtypes(include='number').corr()

        plt.figure(figsize=(12,10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        image_b64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()

        context['plot_b64'] = image_b64


    elif tool_name == 'viz_pairplot':
        import matplotlib
        matplotlib.use('Agg') # Use the non-interactive 'Agg' backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        context['partial_template_name'] = 'edaengine/analysis_partials/_viz_pairplot.html'

        try:
            csv_b64 = request.session.get('csv_base64')
            csv_bytes = base64.b64decode(csv_b64)
            df = pd.read_csv(io.BytesIO(csv_bytes))

            numerical_df = df.select_dtypes(include=['number'])

            if numerical_df.shape[1] >= 2:
                pair_plot_fig = sns.pairplot(numerical_df)

                buffer = io.BytesIO()
                pair_plot_fig.savefig(buffer, format='png')
                buffer.seek(0)
                
                image_b64 = base64.b64encode(buffer.read()).decode('utf-8')
                plt.close(pair_plot_fig.fig)

                context['plot_b64'] = image_b64
            
            else:
                context['plot_b64'] = None
            
        except Exception as e:
            print(f"Error generating pair plot: {e}")
            context['plot_b64'] = None


    elif tool_name == 'analysis_outliers':
        context['partial_template_name'] = 'edaengine/analysis_partials/_outliers.html'

        if request.method == 'POST':
            current_b64 = request.session.get('csv_base64')
            request.session['previous_csv_base64'] = current_b64
            
            action = request.POST.get('action')
            csv_bytes = base64.b64decode(current_b64)
            df = pd.read_csv(io.BytesIO(csv_bytes))

            numerical_cols = df.select_dtypes(include=['number']).columns

            for col in numerical_cols:
                if action == 'cap_99':
                    upper_limit = df[col].quantile(0.99)
                    lower_limit = df[col].quantile(0.01)
                    df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
                
                elif action == 'cap_95':
                    upper_limit = df[col].quantile(0.95)
                    lower_limit = df[col].quantile(0.05)
                    df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)

                elif action == 'trim_1':
                    upper_limit = df[col].quantile(0.99)
                    lower_limit = df.quantile(0.01)
                    df = df[(df[col] >= lower_limit) & (df[col] <= upper_limit)]

            update_session_with_df(request, df)

            request.session['flash_message'] = f"Action '{action}' applied successfully."

            return redirect('analysis_tool', tool_name='analysis_outliers')


    elif tool_name == 'analysis_summary':
        import matplotlib
        matplotlib.use('Agg') # Use the non-interactive 'Agg' backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        context['partial_template_name'] = 'edaengine/analysis_partials/_summary.html'

        csv_b64 = request.session.get('csv_base64')
        csv_bytes = base64.b64decode(csv_b64)
        df = pd.read_csv(io.BytesIO(csv_bytes))

        context['shape'] = df.shape
        context['missing_values'] = df.isnull().sum().to_frame('missing_count').to_html(classes="table table-sm table-bordered")
        context['table'] = df.head().to_html(classes="table table-sm table-bordered")
        context['descriptive_stats'] = df.describe().to_html(classes="table table-sm table-bordered dataframe")

        report_plots = {}

        try:
            corr = df.select_dtypes(include='number').corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            report_plots['correlation_heatmap'] = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
        except Exception as e:
            print(f"Could not generate heatmap for report: {e}")

        numerical_cols = df.select_dtypes(include=['number']).columns
        dist_plots = []
        for col in numerical_cols: # Just take the first two to keep the report concise
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col], kde=True, bins=30)
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            dist_plots.append(base64.b64encode(buffer.read()).decode('utf-8'))
            plt.close()
        report_plots['distributions'] = dist_plots

        context['report_plots'] = report_plots

        if request.method == 'POST' and 'download_pdf' in request.POST:
            template = get_template('edaengine/analysis_partials/_summary.html')
            html = template.render(context)
            result = io.BytesIO()
            pdf = pisa.pisaDocument(io.StringIO(html), result)

            if not pdf.err:
                response = HttpResponse(result.getvalue(), content_type='application.pdf')
                response['Content-Disposition'] = 'attachment; filename="eda_summary_report.pdf"'
                return response

            else:
                return HttpResponse("Error Generating PDF", status=500)

    return render(request, "edaengine/analysis.html", context)



# Feedback Page
# views.py
def feedback_page(request):
    if request.method == 'POST':
        # Get the form data
        name = request.POST.get('name')
        email = request.POST.get('email')
        rating = request.POST.get('rating')
        comments = request.POST.get('comments')

        # For now, just print it to the console
        print("\n--- FEEDBACK RECEIVED ---")
        print(f"Name: {name}")
        print(f"Email: {email}")
        print(f"Rating: {rating}")
        print(f"Comments: {comments}")
        print("-------------------------\n")

        # Set a success message and render the page again
        # Or you could redirect to a simple "thank you" confirmation
        context = {
            'message': "Thank you for your valuable feedback!"
        }
        return render(request, "edaengine/feedback.html", context)

    # For a GET request, just show the blank form
    return render(request, "edaengine/feedback.html")

from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_file, name='upload'),
    path('analysis/', views.analysis, name='analysis_home'),
    path('analysis/<str:tool_name>/', views.analysis, name='analysis_tool'),
    path('action/undo/', views.action_undo, name='action_undo'),
    path('action/download_csv/', views.action_download_csv, name='action_download_csv'),
    path('feedback/', views.feedback_page, name='feedback'),
]

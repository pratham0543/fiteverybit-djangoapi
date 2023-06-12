from django.urls import path
from . import views

urlpatterns = [
    path('shoulder1/', views.shoulder_estimation1, name='shoulder_flexion'),
    path('shoulder2/', views.shoulder_estimation2, name='shoulder_external_rotation'),
    path('leftknee/', views.left_knee_extension, name='left_knee_extension'),
    path('rightknee/', views.right_knee_extension, name='right_knee_extension'),
    path('leftelbow1/', views.left_elbow_flexicon, name='left_elbow_flexion'),
    path('rightelbow1/', views.right_elbow_flexicon, name='right_elbow_flexion'),
    path('leftelbow2/', views.left_elbow_extension, name='left_elbow_extension'),
    path('rightelbow2/', views.right_elbow_extension, name='right_elbow_extension'),
    path('leftankle1/', views.left_ankle_dorsiflexion, name='left_ankle_dorsiflexion'),
    path('rightankle1/', views.right_ankle_dorsiflexion, name='right_ankle_dorsiflexion'),
    path('leftankle2/', views.left_ankle_plantar_flexion, name='left_ankle_plantar_flexion'),
    path('rightankle2/', views.right_ankle_plantar_flexion, name='right_ankle_plantar_flexion')
]


import streamlit as st



st.title('Animal Behavior Tracking')
st.sidebar.title('App Mode')


app_mode = st.sidebar.selectbox('Choose the App Mode',
                                ['About App'])

if app_mode == 'About App':
    st.subheader("About")
    st.markdown("<h5>This is the Animal detection, tracking and behavior recognition App</h5>", unsafe_allow_html=True)
    st.image("Images/00006.jpg")

    # Add demo video section
    st.markdown("<h5>Demo Video</h5>", unsafe_allow_html=True)
    demo_video_path = "output_fixed.mp4"  # Update this path to your actual demo video
    try:
        st.video(demo_video_path)
    except Exception as e:
        st.warning(f"Could not load demo video: {e}")


    st.markdown("""
## Interactive interface installation
(https://github.com/yzxhao123/Behavior_analysis.git)
## Tech Stack
- Python
- PyTorch
- Python CV
- Streamlit
- Yolo
## Reference
[![twitter](https://img.shields.io/badge/Github-1DA1F2?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AntroSafin)
""")

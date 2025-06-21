import { useState } from 'react';
import { Upload, Button, Typography, Space, message, Layout, Divider } from 'antd';
import { UploadOutlined, DownloadOutlined, PlayCircleOutlined, TranslationOutlined } from '@ant-design/icons';
import axios from 'axios';
import Lottie from 'lottie-react';
import lofiAnimation from './anime1.json';

const { Title } = Typography;
const { Header, Content, Footer } = Layout;

function App() {
  const [video, setVideo] = useState(null);
  const [transcribing, setTranscribing] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [englishSubtitle, setEnglishSubtitle] = useState("");
  const [sinhalaSubtitle, setSinhalaSubtitle] = useState("");
  const [englishSubLink, setEnglishSubLink] = useState("");
  const [sinhalaSubLink, setSinhalaSubLink] = useState("");
  const [messageApi, contextHolder] = message.useMessage();

  const props = {
    beforeUpload: file => {
      setVideo(file);
      setTranscript("");
      setEnglishSubtitle("");
      setSinhalaSubtitle("");
      setEnglishSubLink("");
      setSinhalaSubLink("");
      return false;
    },
    maxCount: 1,
  };

  const handleTranscribe = async () => {
    if (!video) {
      messageApi.warning('Please select a video first');
      return;
    }
    const formData = new FormData();
    formData.append('file', video);

    setTranscribing(true);
    setTranscript("");
    try {
      const { data } = await axios.post(
        'http://localhost:8000/transcribe',
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      console.log('Transcription:', data.transcription);
      setTranscript(data.transcription);
      messageApi.success('Transcription successful!');
    } catch (err) {
      console.error(err);
      messageApi.error('Transcription failed.');
    } finally {
      setTranscribing(false);
    }
  };

  const handleGenerate = async () => {
    const formData = new FormData();
    formData.append('text', transcript);

    try {

      const { data } = await axios.post(
        'http://localhost:8000/seg_pun_restoration',
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      setEnglishSubtitle(data.en_script);
      const url = window.URL.createObjectURL(new Blob([data.en_script]));
      setEnglishSubLink(url);
      messageApi.success('English subtitles generated.');
    } catch (err) {
      messageApi.error('Subtitle generation failed.');
    }
  };

  const handleTranslate = async () => {

    const formData = new FormData();
    formData.append('text', englishSubtitle);

    try {
      const { data } = await axios.post(
        'http://localhost:8000/translate',
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      setSinhalaSubtitle(data.sin_script);
      const url = window.URL.createObjectURL(new Blob([data.sin_script]));
      setSinhalaSubLink(url);
      messageApi.success('Translated to Sinhala.');
    } catch (err) {
      messageApi.error('Translation failed.');
    }
  };

  const renderBox = (title, content, backgroundColor = '#fafafa') => (
    <div style={{
      width: '100%',
      maxWidth: 500,
      margin: '16px auto 0',
      border: '1px solid #d9d9d9',
      borderRadius: 8,
      backgroundColor,
      padding: '10px 10px 30px 10px',
      maxHeight: 180,
      overflowY: 'auto',
      overflowX: 'hidden',
      fontSize: 14,
      color: '#333',
      boxShadow: '0 2px 6px rgba(0, 0, 0, 0.05)',
      textAlign: 'center',
      wordWrap: 'break-word',
    }}>
      <Title level={5} style={{ marginTop: 10, marginBottom: 10 }}>{title}</Title>
      <div style={{ whiteSpace: 'pre-wrap' }}>{content}</div>
    </div>
  );

  return (
    <Layout style={{ minHeight: '100vh' }}>
      {contextHolder}
      <Header style={{
        background: '#001529',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        zIndex: 1,
        paddingTop:'12px',
        paddingBottom:'12px',
      }}>
        <Title style={{ color: 'white', margin: 0 }} level={3}>
          🎥 Subtitle Generator & Translator
        </Title>
      </Header>

      <Content style={{
        padding: '40px 80px',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        flexDirection: 'column',
        backgroundColor: '#f0f2f5',
      }}>
        <div style={{
          backgroundColor: 'white',
          padding: '40px 30px 40px',
          borderRadius: '16px',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
          maxWidth: 600,
          width: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          overflow: 'auto',
        }}>
          <Lottie animationData={lofiAnimation} loop={true} style={{ width: 300, marginBottom: 40 }} />

          <Space direction="vertical" size="large" style={{
            width: '100%',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center'
          }}>

            <Upload {...props}>
              <Button icon={<UploadOutlined />}>Select Video</Button>
            </Upload>

            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={handleTranscribe}
              loading={transcribing}
              disabled={!video}
            >
              Upload Video
            </Button>

            {transcript && renderBox("Transcription", transcript)}

            {transcript && (
              <>
            <Divider
              orientation="center"
              style={{
                borderTop: '1px solid #d9d9d9',
              }}
            >
              Subtitle Actions
            </Divider>

                <Button
                  type="dashed"
                  onClick={handleGenerate}
                  icon={<DownloadOutlined />}
                >
                  Generate English Subtitles
                </Button>

                {englishSubLink && (
                  <>
              
                    {englishSubtitle && renderBox("English Subtitles", englishSubtitle)}

                         <a href={englishSubLink} download="subtitles_en.txt">
                      <Button type="link" icon={<DownloadOutlined />}>
                        Download English Subtitles
                      </Button>
                    </a>


                    <Button
                      type="default"
                      onClick={handleTranslate}
                      icon={<TranslationOutlined />}
                    >
                      Translate to Sinhala
                    </Button>
                  </>
                )}

                {sinhalaSubLink && (
                  <>

                    {sinhalaSubtitle && renderBox("Sinhala Subtitles", sinhalaSubtitle)}

                       <a href={sinhalaSubLink} download="subtitles_si.txt">
                      <Button type="link" icon={<DownloadOutlined />}>
                        Download Sinhala Subtitles
                      </Button>
                    </a>
                  </>
                )}
              </>
            )}
          </Space>
        </div>
      </Content>

      <Footer style={{
        textAlign: 'center',
        position: 'fixed',
        bottom: 0,
        left: 0,
        width: '100%',
        background: '#001529',
        zIndex: 1,
        paddingTop:'12px',
        paddingBottom:'12px',
        color:'#fff'
      }}>
        © {new Date().getFullYear()} - CM4340-Natural Language Processing - Group 12
      </Footer>
    </Layout>
  );
}

export default App;

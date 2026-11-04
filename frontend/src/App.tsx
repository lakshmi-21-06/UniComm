import Avatar2D from "./components/Avatar2D";
import { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import {
  Activity,
  AudioLines,
  Camera,
  Languages,
  MessageSquareText,
  Database,
  Gauge,
  Mic,
  Play,
  RefreshCw,
  Send,
  ShieldCheck,
  Sparkles,
  Volume2,
  Video,
  Waves,
  UserRound,
  CheckCircle2,
  AlertCircle,
  FileVideo,
  X,
  LoaderCircle,
  LayoutDashboard,
} from 'lucide-react';

const API = 'http://127.0.0.1:8000/api';

type Tab =
  | 'Dashboard'
  | 'Text Translation'
  | 'Text-to-Speech'
  | 'Speech-to-Text'
  | 'Live Sign Recognition'
  | 'Video Sign Recognition'
  | 'Avatar Communication'
  | 'Dataset Management'
  | 'Training & Evaluation'
  | 'About';

function readableLabel(value: unknown) {
  if (typeof value !== 'string' || !value.trim()) {
    return 'Unknown sign';
  }

  return value
    .replace(/^\d+\./, '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, char => char.toUpperCase());
}
function getSpeechLanguageCode(language: string): 'en' | 'te' | 'hi' {
  if (language.startsWith('te')) return 'te';
  if (language.startsWith('hi')) return 'hi';
  return 'en';
}

export default function App() {
  const [tab, setTab] = useState<Tab>('Dashboard');
  const [health, setHealth] = useState<any>(null);
  const [text, setText] = useState('');
  const [avatarText, setAvatarText] = useState('');
  const [avatarPlayText, setAvatarPlayText] = useState('');
  const [ttsText, setTtsText] = useState('');
  const [output, setOutput] = useState('');
  const [source, setSource] = useState('eng_Latn');
  const [target, setTarget] = useState('tel_Telu');
  const [speechLanguage, setSpeechLanguage] = useState('en-US');
  const [liveTranscribing, setLiveTranscribing] = useState(false);
  const liveSessionRef = useRef(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [amritaPath, setAmritaPath] = useState('');
  const [cslrtPath, setCslrtPath] = useState('');
  const [inventory, setInventory] = useState<any>({});
  const [training, setTraining] = useState<any>(null);
  const [epochs, setEpochs] = useState(10);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoResult, setVideoResult] = useState<any>(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraStatus, setCameraStatus] = useState('Camera is off');
const [recording, setRecording] = useState(false);
const [transcribing, setTranscribing] = useState(false);
const [audioFile, setAudioFile] = useState<File | null>(null);
const [liveResult, setLiveResult] = useState<any>(null);
const [livePredicting, setLivePredicting] = useState(false);
const [liveError, setLiveError] = useState('');
const [handLandmarks, setHandLandmarks] = useState<any[]>([]);
const [handTrackingStatus, setHandTrackingStatus] = useState(
  'Hand tracking is waiting for the camera.'
);
const handCanvasRef = useRef<HTMLCanvasElement>(null);
const liveRecorderRef = useRef<MediaRecorder | null>(null);
const liveChunksRef = useRef<Blob[]>([]);
const liveTimerRef = useRef<number | null>(null);

const liveSessionIdRef = useRef(
  `session-${Date.now()}-${Math.random().toString(36).slice(2)}`
);

const liveFrameTimerRef = useRef<number | null>(null);
const liveFrameBusyRef = useRef(false);
const mediaRecorderRef = useRef<MediaRecorder | null>(null);
const audioChunksRef = useRef<Blob[]>([]);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const cameraOnRef = useRef(false);
  const cameraStartingRef = useRef(false);

  const refresh = async () => {
    try {
      setHealth((await axios.get(`${API}/health`)).data);
      setMessage('Backend connected');
    } catch {
      setHealth(null);
      setMessage('Backend offline — start scripts/start_backend.bat');
    }
  };

  
useEffect(() => {
  const video = videoRef.current;
  const stream = streamRef.current;

  if (cameraOn && video && stream && video.srcObject !== stream) {
    video.srcObject = stream;
    video.play().catch(error => {
      console.error('Could not start webcam preview:', error);
      setCameraStatus('Camera stream attached, but preview playback failed.');
    });
  }
}, [cameraOn, tab]);

useEffect(() => {
  if (!cameraOn || tab !== 'Live Sign Recognition') {
    setHandLandmarks([]);
    setHandTrackingStatus('Hand tracking is waiting for the camera.');
    return;
  }

  let cancelled = false;
  let timeoutId: number | undefined;
  let busy = false;

  const track = async () => {
    if (cancelled || busy) return;

    busy = true;
    try {
      await trackHands();
    } finally {
      busy = false;

      if (!cancelled) {
        timeoutId = window.setTimeout(track, 400);
      }
    }
  };

  void track();

  return () => {
    cancelled = true;
    if (timeoutId !== undefined) {
      window.clearTimeout(timeoutId);
    }
  };
}, [cameraOn, tab]);

useEffect(() => {
  const canvas = handCanvasRef.current;
  const video = videoRef.current;

  if (!canvas || !video) return;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const width = video.videoWidth;
  const height = video.videoHeight;

  if (!width || !height) return;

  canvas.width = width;
  canvas.height = height;

  ctx.clearRect(0, 0, width, height);

  for (const hand of handLandmarks) {
    const points = hand.landmarks || [];

    // Draw the detected hand landmark points.
    ctx.fillStyle = '#00ff88';

    for (const point of points) {
      const x = point.x * width;
      const y = point.y * height;

      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Connect the points to make the hand structure easier to see.
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4],
      [0, 5], [5, 6], [6, 7], [7, 8],
      [5, 9], [9, 10], [10, 11], [11, 12],
      [9, 13], [13, 14], [14, 15], [15, 16],
      [13, 17], [17, 18], [18, 19], [19, 20],
      [0, 17]
    ];

    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 2;

    for (const [start, end] of connections) {
      if (!points[start] || !points[end]) continue;

      ctx.beginPath();
      ctx.moveTo(points[start].x * width, points[start].y * height);
      ctx.lineTo(points[end].x * width, points[end].y * height);
      ctx.stroke();
    }
  }
}, [handLandmarks, cameraOn, tab]);



  const translate = async () => {
    if (!text.trim()) return;

    setBusy(true);
    try {
      const r = await axios.post(`${API}/translate`, {
        text,
        source_language: source,
        target_language: target
      });

      setOutput(r.data.text);
      setMessage(`Translated with ${r.data.model}`);
    } catch (e: any) {
      setMessage(
        e?.response?.data?.detail ||
        'Translation unavailable; check optional model dependencies.'
      );
    } finally {
      setBusy(false);
    }
  };

  
const speak = async () => {
  const value = (
    tab === 'Text-to-Speech'
      ? ttsText
      : output || text
  ).trim();

  if (!value) {
    setMessage('Please enter some text first.');
    return;
  }

  const lang =
    tab === 'Text-to-Speech'
      ? speechLanguage === 'te-IN'
        ? 'te'
        : speechLanguage === 'hi-IN'
        ? 'hi'
        : 'en'
      : target === 'tel_Telu'
      ? 'te'
      : target === 'hin_Deva'
      ? 'hi'
      : 'en';

  const browserLang =
    lang === 'te'
      ? 'te-IN'
      : lang === 'hi'
      ? 'hi-IN'
      : 'en-US';

  try {
    const r = await axios.post(
      `${API}/tts`,
      { text: value, language: lang },
      { responseType: 'blob' }
    );

    const audioUrl = URL.createObjectURL(r.data);
    const audio = new Audio(audioUrl);

    audio.onended = () => URL.revokeObjectURL(audioUrl);
    audio.onerror = () => URL.revokeObjectURL(audioUrl);

    await audio.play();
    setMessage('Playing speech...');
  } catch (error) {
    if ('speechSynthesis' in window) {
      speechSynthesis.cancel();

      const utterance = new SpeechSynthesisUtterance(value);
      utterance.lang = browserLang;

      utterance.onstart = () => setMessage('Playing speech...');
      utterance.onerror = () =>
        setMessage('Browser speech failed. Please try another language.');

      speechSynthesis.speak(utterance);
    } else {
      setMessage('No text-to-speech engine is available.');
    }
  }
};
const handleAudioUpload = async () => {
  if (!audioFile) {
    alert("Please select an audio file first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", audioFile);

  // Convert the selected language to the backend's expected code.
  const language = getSpeechLanguageCode(speechLanguage);
  formData.append("language", language);

console.log("Audio upload language:", language);  
  

  try {
    setTranscribing(true);

    const response = await fetch(`${API}/speech/transcribe`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Audio transcription failed.");
    }

    const result = await response.json();

    setText((previousText) =>
      previousText
        ? `${previousText}\n${result.text}`
        : result.text
    );
  } catch (error) {
    console.error(error);
    alert("Could not transcribe the audio. Please try another file.");
  } finally {
    setTranscribing(false);
  }
};


const startLiveTranscription = async () => {
  if (
    !navigator.mediaDevices?.getUserMedia ||
    !window.MediaRecorder
  ) {
    alert("Your browser does not support audio recording.");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: true,
    });

    const mimeType = MediaRecorder.isTypeSupported(
      "audio/webm;codecs=opus"
    )
      ? "audio/webm;codecs=opus"
      : "audio/webm";

    const recorder = new MediaRecorder(stream, { mimeType });

    liveChunksRef.current = [];
    liveRecorderRef.current = recorder;
    liveSessionRef.current = true;

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        liveChunksRef.current.push(event.data);
      }
    };

    recorder.onstop = async () => {
      const audioBlob = new Blob(liveChunksRef.current, {
        type: recorder.mimeType || "audio/webm",
      });

      liveChunksRef.current = [];

      try {
        if (audioBlob.size === 0) {
          alert("No audio was recorded. Please try again.");
          return;
        }

        const formData = new FormData();
        formData.append("file", audioBlob, "complete-recording.webm");
        const language = getSpeechLanguageCode(speechLanguage);
        formData.append("language", language);

        console.log("Live transcription language:", language);

        const response = await fetch(
          `${API}/speech/transcribe`,
          {
            method: "POST",
            body: formData,
          }
        );

        if (!response.ok) {
          throw new Error(
            `Transcription failed: ${response.status}`
          );
        }

        const result = await response.json();
        const transcript = result.text?.trim();

        if (transcript) {
          // Add the complete transcript only once.
          setText((previousText) =>
            previousText
              ? `${previousText}\n${transcript}`
              : transcript
          );
        } else {
          alert("No speech was recognized. Please try again.");
        }
      } catch (error) {
        console.error("Transcription error:", error);
        alert("Transcription failed. Check the backend terminal.");
      } finally {
        stream.getTracks().forEach((track) => track.stop());
        liveRecorderRef.current = null;
        setLiveTranscribing(false);
      }
    };

    recorder.start();

    setLiveTranscribing(true);
  } catch (error) {
    console.error("Microphone error:", error);
    alert("Could not access your microphone. Check microphone permissions.");
    setLiveTranscribing(false);
  }
};

const stopLiveTranscription = () => {
  liveSessionRef.current = false;

  const recorder = liveRecorderRef.current;

  if (recorder && recorder.state !== "inactive") {
    recorder.stop();
  } else {
    setLiveTranscribing(false);
  }
};

  const startRecording = async () => {
  try {
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      setMessage('Microphone recording is not supported by this browser.');
      return;
    }

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/mp4'
    ].find(type => MediaRecorder.isTypeSupported(type));

    const recorder = mimeType
      ? new MediaRecorder(stream, { mimeType })
      : new MediaRecorder(stream);

    audioChunksRef.current = [];

    recorder.ondataavailable = event => {
      if (event.data.size > 0) {
        audioChunksRef.current.push(event.data);
      }
    };

    recorder.onstop = async () => {
      stream.getTracks().forEach(track => track.stop());

      const audioBlob = new Blob(audioChunksRef.current, {
        type: recorder.mimeType || 'audio/webm'
      });

      if (!audioBlob.size) {
        setMessage('No audio was recorded. Please try again.');
        return;
      }

      const extension = audioBlob.type.includes('mp4') ? 'mp4' : 'webm';
      const formData = new FormData();
      formData.append('file', audioBlob, `recording.${extension}`);
      
      const languageCode = getSpeechLanguageCode(speechLanguage);
      formData.append('language', languageCode);

      console.log("Microphone recording language:", languageCode);


      setTranscribing(true);
      setMessage('Transcribing your recording…');

      try {
        const response = await axios.post(
          `${API}/speech/transcribe`,
          formData,
          { timeout: 180000 }
        );

        const recognizedText = String(response.data.text || '').trim();

        if (recognizedText) {
          setText(previous =>
            previous.trim()
              ? `${previous.trim()} ${recognizedText}`
              : recognizedText
          );
          setMessage(
            `Speech recognized (${response.data.language || 'language unknown'}).`
          );
        } else {
          setMessage('No speech was recognized. Try recording again.');
        }
      } catch (error: any) {
        setMessage(
          error?.response?.data?.detail ||
          'Transcription failed. Check that the backend and speech model are available.'
        );
      } finally {
        setTranscribing(false);
      }
    };

    mediaRecorderRef.current = recorder;
    recorder.start();
    setRecording(true);
    setMessage('Recording… Speak clearly, then select Stop Recording.');
  } catch (error: any) {
    setMessage(
      error?.name === 'NotAllowedError'
        ? 'Microphone permission was denied. Allow microphone access in your browser.'
        : 'Could not start microphone recording. Check your microphone connection.'
    );
  }
};

const stopRecording = () => {
  const recorder = mediaRecorderRef.current;

  if (recorder && recorder.state !== 'inactive') {
    recorder.stop();
  }

  setRecording(false);
};

  const inspect = async (dataset: 'amrita' | 'isl_cslrt') => {
    setBusy(true);

    try {
      const path = dataset === 'amrita' ? amritaPath : cslrtPath;

      if (path.trim()) {
        await axios.post(
          `${API}/datasets/configure`,
          dataset === 'amrita'
            ? { amrita_path: path }
            : { isl_cslrt_path: path }
        );
      }

      const r = await axios.post(`${API}/datasets/inspect`, { dataset });

      setInventory((old: any) => ({ ...old, [dataset]: r.data }));
      setMessage(`${dataset} inspection completed`);
    } catch (e: any) {
      setMessage(e?.response?.data?.detail || 'Dataset inspection failed.');
    } finally {
      setBusy(false);
    }
  };

  const preprocess = async () => {
    setBusy(true);

    try {
      const r = await axios.post(`${API}/datasets/preprocess/amrita`);

      setInventory((old: any) => ({ ...old, preprocess: r.data }));
      setMessage(
        `Processed ${r.data.successful_videos} videos; ${r.data.failed_videos} failed.`
      );
    } catch (e: any) {
      setMessage(e?.response?.data?.detail || 'Preprocessing failed.');
    } finally {
      setBusy(false);
    }
  };

  const train = async () => {
    setBusy(true);

    try {
      const r = await axios.post(`${API}/train`, {
        dataset: 'amrita',
        epochs,
        batch_size: 16,
        learning_rate: 0.0005,
        sequence_length: 32
      });

      setTraining({ status: 'completed', metrics: r.data });
      setMessage('Training and evaluation completed.');
    } catch (e: any) {
      setMessage(e?.response?.data?.detail || 'Training failed.');
    } finally {
      setBusy(false);
    }
  };


const evaluate = async () => {
  try {
    const r = await axios.get(`${API}/evaluation`);

    if (r.data.available) {
      setTraining((old: any) => ({
        ...(old || {}),
        metrics: {
          test_accuracy: r.data.test_accuracy,
          macro_precision: r.data.macro_precision,
          macro_recall: r.data.macro_recall,
          macro_f1: r.data.macro_f1,
        },
        evaluation: r.data,
      }));

      setMessage('Saved evaluation loaded successfully.');
    } else {
      setMessage(r.data.message || 'No saved evaluation is available.');
    }
  } catch (error) {
    console.error('Evaluation loading error:', error);
    setMessage('Could not load evaluation. Check that the backend is running.');
  }
};


  const predict = async () => {
    if (!videoFile) return;

    setBusy(true);
    setVideoResult(null);

    try {
      const form = new FormData();
      form.append('file', videoFile);

      const r = await axios.post(`${API}/sign/predict-video`, form);
      setVideoResult(r.data);
    } catch (e: any) {
      setVideoResult({
        error:
          e?.response?.data?.detail ||
          'Prediction failed. Check that the backend is running and the video is supported.'
      });
    } finally {
      setBusy(false);
    }
  };
  


const startLiveRecognition = () => {
  const stream = streamRef.current;
  const video = videoRef.current;

  if (
    !stream ||
    !video ||
    !stream.getVideoTracks().some(track => track.readyState === 'live') ||
    liveFrameBusyRef.current
  ) {
    return;
  }

  const stillRunning = () =>
    streamRef.current === stream &&
    stream.getVideoTracks().some(track => track.readyState === 'live') &&
    video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    video.videoWidth > 0 &&
    video.videoHeight > 0;

  const sendNextFrame = async () => {
    if (!stillRunning()) return;

    // Wait until the previous request finishes to avoid overlapping requests.
    if (liveFrameBusyRef.current) {
      liveFrameTimerRef.current = window.setTimeout(sendNextFrame, 150);
      return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext('2d');

    if (!context) {
      setLiveError('Could not process the webcam frame.');
      return;
    }

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    liveFrameBusyRef.current = true;
    setLivePredicting(true);
    setLiveError('');

    try {
      const blob = await new Promise<Blob | null>(resolve => {
        canvas.toBlob(resolve, 'image/jpeg', 0.85);
      });

      if (!blob || !stillRunning()) return;

      const form = new FormData();
      form.append('file', blob, 'webcam-frame.jpg');
      form.append('session_id', liveSessionIdRef.current);

      const response = await axios.post(
        `${API}/sign/live-frame`,
        form,
        { timeout: 20000 }
      );

      if (!stillRunning()) return;

      const data = response.data;

      // Update the displayed result only when the backend has a prediction.
      // This prevents temporary "collecting frames" responses from erasing it.
      if (data?.label || data?.predicted_class) {
        setLiveResult(data);
      } else if (data?.detected === false && data?.ready === false) {
        setLiveResult(null);
      }

      if (data?.message) {
        setHandTrackingStatus(data.message);
      }
    } catch (error: any) {
      if (stillRunning()) {
        setLiveError(
          error?.response?.data?.detail ||
          'Live prediction failed. Check the backend and trained model.'
        );
      }
    } finally {
      liveFrameBusyRef.current = false;
      setLivePredicting(false);
    }

    // Send the next frame after a short pause.
    if (stillRunning()) {
      liveFrameTimerRef.current = window.setTimeout(sendNextFrame, 150);
    }
  };

  void sendNextFrame();
};


const trackHands = async () => {
  const video = videoRef.current;

  if (
    !video ||
    video.readyState < 2 ||
    video.videoWidth === 0 ||
    video.videoHeight === 0
  ) {
    return;
  }

  try {
    const frameCanvas = document.createElement('canvas');
    frameCanvas.width = video.videoWidth;
    frameCanvas.height = video.videoHeight;

    const context = frameCanvas.getContext('2d');
    if (!context) return;

    context.drawImage(video, 0, 0);

    const blob = await new Promise<Blob | null>(resolve =>
      frameCanvas.toBlob(resolve, 'image/jpeg', 0.8)
    );

    if (!blob) return;

    const form = new FormData();
    form.append('file', blob, 'webcam.jpg');

    const response = await axios.post(
      `${API}/sign/hand-landmarks-image`,
      form
    );

    setHandLandmarks(response.data.hands || []);
    setHandTrackingStatus(
      response.data.detected
        ? 'Hands detected'
        : 'No hands detected — keep both hands visible'
    );
  } catch (error) {
    console.error('Hand tracking failed:', error);
    setHandTrackingStatus('Hand tracking unavailable');
  }
};


const camera = async () => {
  // Prevent duplicate start requests.
  if (cameraStartingRef.current) return;

  // STOP CAMERA
  if (cameraOnRef.current) {
    cameraOnRef.current = false;
    cameraStartingRef.current = false;

    // Cancel scheduled frame recognition.
    if (liveFrameTimerRef.current !== null) {
      window.clearTimeout(liveFrameTimerRef.current);
      liveFrameTimerRef.current = null;
    }

    if (liveTimerRef.current !== null) {
      window.clearTimeout(liveTimerRef.current);
      liveTimerRef.current = null;
    }

    // Stop any old video recorder.
    const recorder = liveRecorderRef.current;

    if (recorder && recorder.state !== 'inactive') {
      recorder.stop();
    }

    liveRecorderRef.current = null;

    // Stop every camera track.
    const stream = streamRef.current;
    streamRef.current = null;

    if (stream) {
      stream.getTracks().forEach(track => {
        track.stop();
      });
    }

    // Detach the video element.
    const video = videoRef.current;

    if (video) {
      video.pause();
      video.srcObject = null;
    }

    liveFrameBusyRef.current = false;

    setCameraOn(false);
    setCameraStatus('Camera is off');
    setLivePredicting(false);
    setLiveResult(null);
    setLiveError('');
    setHandLandmarks([]);
    setHandTrackingStatus('Hand tracking is waiting for the camera.');

    return;
  }

  // START CAMERA
  cameraStartingRef.current = true;
  let stream: MediaStream | null = null;

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(
        'Camera access is unavailable. Use localhost or HTTPS.'
      );
    }

    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'user',
        width: { ideal: 640 },
        height: { ideal: 480 }
      },
      audio: false
    });

    // The user might have cancelled while permission was pending.
    if (!cameraStartingRef.current) {
      stream.getTracks().forEach(track => track.stop());
      return;
    }

    streamRef.current = stream;
    cameraOnRef.current = true;

    setLiveResult(null);
    setLiveError('');
    setLivePredicting(false);
    setCameraStatus('Opening webcam preview...');
    setCameraOn(true);

    // Wait for React to render the video element.
    await new Promise<void>(resolve => {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => resolve());
      });
    });

    if (!cameraOnRef.current || streamRef.current !== stream) {
      stream.getTracks().forEach(track => track.stop());
      return;
    }

    const video = videoRef.current;

    if (!video) {
      throw new Error('Webcam video element was not found.');
    }

    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;

    await video.play();

    if (!cameraOnRef.current || streamRef.current !== stream) {
      return;
    }

    if (video.videoWidth === 0 || video.videoHeight === 0) {
      throw new Error('The webcam opened but no video frames are available.');
    }

    liveSessionIdRef.current =
      `session-${Date.now()}-${Math.random().toString(36).slice(2)}`;

    setCameraStatus('Camera active. Recognizing signs...');
    setHandTrackingStatus('Camera active. Show a supported sign clearly.');

    startLiveRecognition();
  } catch (error: any) {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
    }

    if (streamRef.current === stream) {
      streamRef.current = null;
    }

    cameraOnRef.current = false;
    setCameraOn(false);
    setLivePredicting(false);
    setCameraStatus('Camera is off');

    setLiveError(
      error?.name === 'NotAllowedError'
        ? 'Camera permission denied. Allow camera access in your browser.'
        : error?.name === 'NotFoundError'
          ? 'No webcam was found.'
          : error?.name === 'NotReadableError'
            ? 'The webcam may be in use by another application.'
            : error?.message || 'Could not start the webcam.'
    );
  } finally {
    cameraStartingRef.current = false;
  }
};






  const menu: { name: Tab; icon: any }[] = [
  { name: 'Dashboard', icon: LayoutDashboard },
  { name: 'Text Translation', icon: Languages },
  { name: 'Text-to-Speech', icon: Volume2 },
  { name: 'Speech-to-Text', icon: Mic },
  { name: 'Live Sign Recognition', icon: Camera },
  { name: 'Video Sign Recognition', icon: Video },
  { name: 'Avatar Communication', icon: MessageSquareText },
  { name: 'Dataset Management', icon: Database },
  { name: 'Training & Evaluation', icon: Gauge },
  { name: 'About', icon: Sparkles },
];

  const confidence =
    typeof videoResult?.confidence === 'number' &&
    Number.isFinite(videoResult.confidence)
      ? Math.max(0, Math.min(100, videoResult.confidence * 100))
      : null;

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon"><Waves /></div>
          <div>
            <b>UniComm</b>
            <small>AI COMMUNICATION LAYER</small>
          </div>
        </div>

        <p className="nav-label">WORKSPACE</p>

        {menu.map(item => (
          <button
            key={item.name}
            className={`nav ${tab === item.name ? 'active' : ''}`}
            onClick={() => setTab(item.name)}
          >
            <item.icon size={18} />
            {item.name}
          </button>
        ))}

        <div className="side-status">
          <i />
          {health ? 'Backend connected' : 'Backend offline'}
          <button onClick={refresh} aria-label="Refresh backend status">
            <RefreshCw size={14} />
          </button>
        </div>
      </aside>

      <main>
        <header>
          <div>
            <small>UNIVERSAL ACCESSIBILITY PLATFORM</small>
            <h1>{tab}</h1>
          </div>
          <span className={health ? 'online' : 'offline'}>
            {health ? '● SYSTEM ONLINE' : '● API OFFLINE'}
          </span>
        </header>

        {message && (
          <div className="notice" role="status">
            <Activity size={16} />
            {message}
            <button onClick={() => setMessage('')} aria-label="Dismiss message">
              <X size={15} />
            </button>
          </div>
        )}

        {tab === 'Dashboard' && (
  <>
    <section className="hero">
      <div>
        <small>WELCOME TO UNICOMM</small>
        <h2>
          Communication for<br />
          <em>everyone.</em>
        </h2>
        <p>
          Connect through text, speech, and sign language.
          Choose a communication tool to get started.
        </p>
      </div>
    </section>

    <section className="dashboard-features">
      <button
        className="feature-card"
        onClick={() => setTab('Text Translation')}
      >
        <Languages size={28} />
        <h3>Text Translation</h3>
        <p>Translate text between English, Telugu, and Hindi.</p>
        <span>Open feature →</span>
      </button>

      <button
        className="feature-card"
        onClick={() => setTab('Text-to-Speech')}
      >
        <Volume2 size={28} />
        <h3>Text-to-Speech</h3>
        <p>Convert written messages into spoken audio.</p>
        <span>Open feature →</span>
      </button>

      <button
        className="feature-card"
        onClick={() => setTab('Speech-to-Text')}
      >
        <Mic size={28} />
        <h3>Speech-to-Text</h3>
        <p>Convert spoken words into written text.</p>
        <span>Open feature →</span>
      </button>

      <button
        className="feature-card"
        onClick={() => setTab('Video Sign Recognition')}
      >
        <Camera size={28} />
        <h3>Sign Recognition</h3>
        <p>Explore sign recognition using supported video input.</p>
        <span>Open feature →</span>
      </button>

      <button
        className="feature-card"
        onClick={() => setTab('Avatar Communication')}
      >
        <UserRound size={28} />
        <h3>Avatar Communication</h3>
        <p>Explore avatar-based sign communication.</p>
        <span>Open feature →</span>
      </button>

      <button
        className="feature-card"
        onClick={() => setTab('Dataset Management')}
      >
        <Database size={28} />
        <h3>Dataset Management</h3>
        <p>Access tools for viewing and preparing datasets.</p>
        <span>Open feature →</span>
      </button>
    </section>
  </>
)}
          
        {tab === 'Text Translation' && (
          <section className="panel">
            <small>LANGUAGE TOOLS</small>
            <h3>Text Translation</h3>
            <p>Translate text between English, Telugu, and Hindi.</p>

            <label htmlFor="translation-input">Enter text to translate</label>
            <textarea
              id="translation-input"
              rows={5}
              value={text}
              onChange={e => setText(e.target.value)}
              placeholder="Type your text here..."
            />

            <div className="controls">
              <div className="lang-pair">
                <label>
                  FROM
                  <select
                    value={source}
                    onChange={e => setSource(e.target.value)}
                  >
                    <option value="eng_Latn">English</option>
                    <option value="tel_Telu">Telugu</option>
                    <option value="hin_Deva">Hindi</option>
                  </select>
                </label>

                <Languages />

                <label>
                  TO
                  <select
                    value={target}
                    onChange={e => setTarget(e.target.value)}
                  >
                    <option value="tel_Telu">Telugu</option>
                    <option value="hin_Deva">Hindi</option>
                    <option value="eng_Latn">English</option>
                  </select>
                </label>
              </div>

              <button
                className="primary"
                disabled={busy || !text.trim()}
                onClick={translate}
              >
                {busy ? 'Translating...' : 'Translate'}
                <Send size={14} />
              </button>
            </div>

            {output && (
              <div className="output">
                <small>TRANSLATED OUTPUT</small>
                <p>{output}</p>

                <button
                  className="secondary"
                  onClick={() => navigator.clipboard?.writeText(output)}
                >
                  Copy translation
                </button>
              </div>
            )}
          </section>
        )}

              
        {tab === 'Text-to-Speech' && (
          <section className="panel">
            <small>VOICE TOOLS</small>
            <h3>Text-to-Speech</h3>
            <p>Convert your text into spoken audio.</p>

            <label htmlFor="tts-input">Enter text to speak</label>
            <textarea
              id="tts-input"
              rows={5}
              value={ttsText}
              onChange={e => setTtsText(e.target.value)}
              placeholder="Type something you want to hear..."
            />

            <label htmlFor="tts-language">Choose language</label>
            <select
              id="tts-language"
              value={speechLanguage}
              onChange={e => setSpeechLanguage(e.target.value)}
            >
              <option value="en-US">English</option>
              <option value="te-IN">Telugu</option>
              <option value="hi-IN">Hindi</option>
            </select>

            <button
              className="primary"
              disabled={busy || !ttsText.trim()}
              onClick={speak}
            >
              <Volume2 size={16} />
              Speak Text
            </button>
          </section>
        )}
        
        {tab === 'Avatar Communication' && (
          <section className="panel">
            <small>AVATAR COMMUNICATION</small>
            <h3>Text-to-Sign Avatar</h3>
            <p>
              Enter text to display the corresponding supported Indian Sign
              Language (ISL) movements using your 3D avatar.
            </p>

            <label htmlFor="avatar-text">
              Enter text for the avatar
            </label>

            <textarea
              id="avatar-text"
              rows={4}
              value={avatarText}
              onChange={e => setAvatarText(e.target.value)}
              placeholder="Enter a word, e.g. Hello"
            />

            <button
              className="primary"
              disabled={!avatarText.trim()}
              onClick={() => setAvatarPlayText(avatarText)}
            >
              <Play size={16} />
              Prepare Avatar
            </button>

            <div style={{ marginTop: 24 }}>
              <Avatar2D text={avatarPlayText} />
            </div>
          </section>
        )}
        


{tab === 'Speech-to-Text' && (
  <section className="panel">
    <small>VOICE TOOLS</small>
    <h3>Speech-to-Text</h3>
    <p>Select the language spoken in your audio, then upload a file or speak into your microphone to convert speech into text.</p>

    <div style={{ marginBottom: 24 }}>
      <label htmlFor="speech-language">Choose spoken language</label>
      <select
        id="speech-language"
        value={speechLanguage}
        onChange={(e) => setSpeechLanguage(e.target.value)}
        disabled={liveTranscribing || transcribing}
      >
        <option value="en-US">English</option>
        <option value="te-IN">Telugu</option>
        <option value="hi-IN">Hindi</option>
      </select>
    </div>

    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
        gap: 20,
        marginBottom: 24,
      }}
    >
      {/* Upload Audio */}
      <div
        style={{
          padding: 20,
          border: "1px solid #263650",
          borderRadius: 12,
          background: "#0d1728",
        }}
      >
        <h4 style={{ marginTop: 0 }}>Upload Audio</h4>
        <p style={{ fontSize: 13 }}>
          Transcribe an existing audio file.
        </p>

        <input
          id="audio-upload"
          type="file"
          accept="audio/*,.wav,.mp3,.m4a,.ogg,.webm"
          onChange={(e) => setAudioFile(e.target.files?.[0] ?? null)}
          disabled={transcribing || liveTranscribing}
        />

        <button
          className="primary"
          onClick={handleAudioUpload}
          disabled={!audioFile || transcribing || liveTranscribing}
          style={{ marginTop: 14 }}
        >
          {transcribing ? "Transcribing..." : "Upload & Transcribe"}
        </button>
      </div>

      {/* Transcription */}
      <div
        style={{
          padding: 20,
          border: "1px solid #263650",
          borderRadius: 12,
          background: "#0d1728",
        }}
      >
        <h4 style={{ marginTop: 0 }}>Transcription</h4>
        <p style={{ fontSize: 13 }}>
          Speak naturally and convert your speech into text.
        </p>

        {!liveTranscribing ? (
          <button
            className="primary"
            onClick={startLiveTranscription}
            disabled={transcribing}
          >
            Start Recording
          </button>
        ) : (
          <button
            className="primary"
            onClick={stopLiveTranscription}
          >
            Stop Recording
          </button>
        )}

        {liveTranscribing && (
          <p style={{ fontSize: 13 }}>
            Listening to your microphone...
          </p>
        )}
      </div>
    </div>

    <label htmlFor="speech-output">Recognized Text</label>
    <textarea
      id="speech-output"
      rows={7}
      value={text}
      onChange={(e) => setText(e.target.value)}
      placeholder="Your recognized speech will appear here..."
      style={{ marginTop: 8 }}
    />
  </section>
)}




  {tab === 'Dataset Management' && (
  <section className="panel dataset-details">
    <small>DATASET MANAGEMENT</small>
    <h3>Amrita SLR Dataset</h3>

    <p>Configure and process your Indian Sign Language dataset.</p>

    <div className="dataset-row">
      <label>
        Dataset Folder

        <input
          value={amritaPath}
          onChange={e => setAmritaPath(e.target.value)}
          placeholder="Enter dataset folder path"
        />
      </label>

      <button
        className="primary"
        disabled={busy}
        onClick={() => inspect('amrita')}
      >
        {busy ? 'Inspecting…' : 'Inspect Dataset'}
      </button>
    </div>

    <div className="dataset-actions">
      <div>
        <h4>Landmark Extraction</h4>
        <p>Extract pose and hand landmarks from dataset videos.</p>
      </div>

      <button
        className="secondary"
        disabled={busy}
        onClick={preprocess}
      >
        {busy ? 'Processing…' : 'Extract Landmarks'}
      </button>
    </div>

    {Object.entries(inventory)
      .filter(([key]) => key === 'amrita')
      .map(([key, value]: any) => (
        <div className="json" key={key}>
          <b>Dataset Inspection Report</b>
          <pre>{JSON.stringify(value, null, 2)}</pre>
        </div>
      ))}
  </section>
)}

        {tab === 'Training & Evaluation' && (
  <section className="training-page">
    <div className="training-intro">
      <small>MODEL LAB / PERFORMANCE</small>
      <h2>Training & Evaluation</h2>
      <p>
        Train your sign recognition model and review its evaluation results.
      </p>
    </div>

    <section className="panel training-panel">
      <div className="panel-title">
        <div>
          <small>MODEL CONFIGURATION</small>
          <h3>Train your recognition model</h3>
          <p>Configure the training run and monitor its status.</p>
        </div>

        <span className="training-badge">
          {training?.running
            ? '● RUNNING'
            : training?.metrics
              ? '● RESULTS AVAILABLE'
              : '● READY'}
        </span>
      </div>

      <div className="warning">
        <AlertCircle size={18} />
        <span>
          The current model uses Dataset 1 (Amrita SLR).
          Verify dataset annotations and mappings before training.
        </span>
      </div>

      <div className="epoch-setting">
        <div>
          <label htmlFor="training-epochs">Training epochs</label>
          <p>Choose how many epochs to run.</p>
        </div>
        <strong>{epochs}</strong>
      </div>

      <input
        id="training-epochs"
        type="range"
        min="1"
        max="50"
        value={epochs}
        onChange={e => setEpochs(Number(e.target.value))}
        disabled={busy || training?.running}
      />

      <div className="training-actions">
        <button
          className="primary"
          disabled={busy || training?.running}
          onClick={train}
        >
          <Play size={16} />
          {training?.running ? 'Training in progress…' : 'Start training'}
        </button>

        <button
          className="secondary"
          disabled={busy}
          onClick={evaluate}
        >
          <Gauge size={16} />
          Load evaluation results
        </button>
      </div>
    </section>
<section className="training-results">
  <div className="results-heading">
    <div>
      <small>MODEL PERFORMANCE</small>
      <h3>Evaluation results</h3>
    </div>
    <span>
      {training?.evaluation?.available || training?.metrics
        ? 'Metrics available'
        : 'Awaiting evaluation'}
    </span>
  </div>

  <div className="metrics">
    {[
      [
        'Test accuracy',
        training?.metrics?.test_accuracy ??
          training?.evaluation?.test_accuracy,
      ],
      [
        'Macro precision',
        training?.metrics?.macro_precision ??
          training?.evaluation?.macro_precision,
      ],
      [
        'Macro recall',
        training?.metrics?.macro_recall ??
          training?.evaluation?.macro_recall,
      ],
      [
        'Macro F1-score',
        training?.metrics?.macro_f1 ??
          training?.evaluation?.macro_f1,
      ],
    ].map(([key, value]: any) => (
      <div className="training-metric" key={key}>
        <small>{key}</small>
        <b>
          {typeof value === 'number'
            ? `${(value * 100).toFixed(2)}%`
            : '—'}
        </b>
        <span>
          {typeof value === 'number'
            ? 'Saved evaluation result'
            : 'No result available'}
        </span>
      </div>
    ))}
  </div>
</section>


    <section className="panel training-status-panel">
      <div className="panel-title">
        <div>
          <small>RUN INFORMATION</small>
          <h3>Training status</h3>
        </div>
      </div>

      <div className="json">
        <pre>
          {JSON.stringify(
            training || { status: 'No training run yet' },
            null,
            2
          )}
        </pre>
      </div>
    </section>
  </section>
)}

        {tab === 'Video Sign Recognition' && (
          <section className="panel video-panel">
            <small>VIDEO RECOGNITION</small>
            <h3>Recognize a supported sign clip</h3>
            <p>
              Upload a video to identify a sign learned by your trained model.
            </p>

            <label className="video-upload">
              <FileVideo size={21} />
              <span>
                {videoFile ? videoFile.name : 'Choose a video file'}
                <small>
                  {videoFile
                    ? `${(videoFile.size / (1024 * 1024)).toFixed(2)} MB`
                    : 'Select a supported video from your computer'}
                </small>
              </span>
              <input
                type="file"
                accept="video/*"
                onChange={e => {
                  setVideoFile(e.target.files?.[0] || null);
                  setVideoResult(null);
                }}
              />
            </label>

            <div className="actions">
              <button
                className="primary"
                disabled={busy || !videoFile}
                onClick={predict}
              >
                {busy
                  ? <><LoaderCircle size={16} /> Analyzing…</>
                  : <><Activity size={16} /> Analyze video</>}
              </button>

              {videoResult && (
                <button
                  className="secondary"
                  onClick={() => setVideoResult(null)}
                >
                  Clear result
                </button>
              )}
            </div>

            {videoResult && (
              <div
                className={`recognition-result ${videoResult.error ? 'result-error' : ''}`}
                aria-live="polite"
              >
                {videoResult.error ? (
                  <>
                    <AlertCircle size={27} />
                    <h3>Recognition unsuccessful</h3>
                    <p>{String(videoResult.error)}</p>
                    <small>Check the video file and backend, then try again.</small>
                  </>
                ) : (
                  <>
                    <div className="result-heading">
                      <CheckCircle2 size={22} />
                      <small>RECOGNITION RESULT</small>
                    </div>

                    <p className="result-caption">Predicted sign</p>
                    <h2 className="predicted-label">
                      {readableLabel(videoResult.label)}
                    </h2>

                    {confidence !== null && (
                      <div className="confidence-section">
                        <div className="confidence-row">
                          <span>Model confidence</span>
                          <b>{confidence.toFixed(2)}%</b>
                        </div>
                        <div
                          className="confidence-track"
                          role="progressbar"
                          aria-label="Model confidence"
                          aria-valuemin={0}
                          aria-valuemax={100}
                          aria-valuenow={confidence}
                        >
                          <div
                            className="confidence-fill"
                            style={{ width: `${confidence}%` }}
                          />
                        </div>
                      </div>
                    )}

                    
                  </>
                )}
              </div>
            )}
          </section>
        )}

        
{tab === 'Live Sign Recognition' && (
  <section className="panel">
    <small>LIVE CAMERA</small>
    <h3>Live sign recognition</h3>
    <p>
      Hold one supported sign clearly in view for a short clip.
      Predictions depend on your trained model and dataset.
    </p>
    
<div
  className="camera"
  style={{ position: 'relative', overflow: 'hidden' }}
>
  {cameraOn ? (
    <>
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        style={{
          display: 'block',
          width: '100%',
          height: '100%',
          objectFit: 'contain',
          background: '#000'
        }}
      />

      <canvas
        ref={handCanvasRef}
        style={{
          position: 'absolute',
          inset: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none'
        }}
      />
    </>
  ) : (
    <div>
      <Camera size={38} />
      <b>Camera preview</b>
      <small>Enable camera access to begin</small>
    </div>
  )}
</div>

<p role="status">{handTrackingStatus}</p>
    <div className="actions">
      <button
      className="primary"
      onClick={camera}
    >
      {cameraOn ? 'Stop camera' : 'Enable webcam'}
    </button>
      <span>{cameraStatus}</span>
    </div>

    {livePredicting && (
      <p role="status">
        <LoaderCircle size={16} /> Recognizing sign…
      </p>
    )}

    {liveError && (
      <div className="warning" role="alert">
        <AlertCircle size={18} />
        {liveError}
      </div>
    )}
    {liveResult && liveResult.detected === false && (
  <div className="warning" role="status">
    {liveResult.message || 'No hand detected — prediction skipped.'}
  </div>
)}

    {liveResult && liveResult.detected && !liveResult.error && (
      <div className="recognition-result" aria-live="polite">
        <div className="result-heading">
          <CheckCircle2 size={22} />
          <small>LATEST PREDICTION</small>
        </div>

        
        {liveResult.accepted === false ? (
          <div className="warning" role="status">
            <h2>Unclear sign</h2>
            <p>Please perform the sign again, keeping your hand clearly visible.</p>
          </div>
        ) : (
          <h2 className="predicted-label">
            {readableLabel(liveResult.label)}
          </h2>
        )}

        {typeof liveResult.confidence === 'number' &&
          Number.isFinite(liveResult.confidence) && (
            <div className="confidence-section">
              <div className="confidence-row">
                <span>Model confidence</span>
                <b>
                  {(Math.max(0, Math.min(1, liveResult.confidence)) * 100)
                    .toFixed(2)}%
                </b>
              </div>
              <div className="confidence-track">
                <div
                  className="confidence-fill"
                  style={{
                    width: `${Math.max(
                      0,
                      Math.min(100, liveResult.confidence * 100)
                    )}%`
                  }}
                />
              </div>
            </div>
          )}

        <small>
          Model output is a prediction, not a guarantee of correct interpretation.
        </small>
      </div>
    )}

    <div className="warning">
      Live recognition collects webcam frames continuously. Keep your hands
      visible, perform one supported sign clearly, and allow a few seconds
      for the model to make a prediction.
    </div>
  </section>
)}


        {tab === 'About' && (
  <section className="panel">
    <small>ABOUT UNICOMM</small>

    <h3>Bringing people closer through accessible communication</h3>

    <p>
      UniComm is an AI-powered communication platform designed to reduce
      communication barriers across languages and communication methods.
      It brings together text translation, speech processing and sign
      language recognition in one unified workspace.
    </p>

    <div className="about-grid">
      <div>
        <b>Platform Features</b>
        <ul>
          <li>Multilingual text translation</li>
          <li>Text-to-speech and speech-to-text conversion</li>
          <li>Live sign language recognition using computer vision</li>
          <li>Video-based sign language recognition</li>
          <li>Animated avatar-based sign communication</li>
          <li>Dataset inspection and landmark extraction</li>
          <li>Model training and evaluation</li>
        </ul>
      </div>

      <div>
        <b>Technology Stack</b>
        <ul>
          <li>React and TypeScript for the frontend</li>
          <li>Python and FastAPI for backend services</li>
          <li>Machine learning and deep learning models</li>
          <li>MediaPipe and computer vision</li>
          <li>Natural language processing</li>
          <li>Speech recognition and speech synthesis</li>
          <li>Three.js-based avatar visualization</li>
        </ul>
      </div>

      <div>
        <b>Project Highlights</b>
        <ul>
          <li>Unified communication workspace</li>
          <li>Support for English, Telugu and Hindi workflows</li>
          <li>AMRITA dataset-based sign recognition workflow</li>
          <li>Configurable dataset paths and inspection</li>
          <li>Integrated translation, speech and sign tools</li>
        </ul>
      </div>

      <div>
      <b>Future Scope</b>
      <ul>
        <li>Introducing sign-to-sentence translation for more natural communication</li>
        <li>Expanding sign language datasets with more diverse recordings</li>
        <li>Developing dynamic datasets to support continuous data updates</li>
        <li>Adding support for more regional and Indian languages</li>
        <li>Improving sign recognition accuracy across different users and environments</li>
        <li>Enhancing avatar animations with a wider vocabulary of signs</li>
      </ul>
    </div>
    </div>
  </section>
)}

        <footer>
          <span>UniComm · Universal AI Communication Layer</span>
          <span>Technology for a more inclusive and connected world.</span>
        </footer>
      </main>
    </div>
  );
}

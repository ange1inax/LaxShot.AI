// app/upload/page.tsx
'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, Film, CheckCircle, AlertCircle, Loader2, Target, TrendingUp, Award, Zap, Wifi, WifiOff, User, BarChart, ChevronDown, ChevronUp } from 'lucide-react';

interface ShotBreakdown {
  shot_number: number;
  timestamp: number;
  shot_type: string;
  metrics: {
    load_angle: number;
    release_angle: number;
    angle_change: number;
    shot_speed: number;
    max_layback: number;
    max_h2ss: number;
    max_hand_velocity: number;
    trunk_rotation: number;
    arm_extension_style: string;
    arm_path_type: string;
  };
  form_score: number;
  feedback: Array<{
    type: string;
    message: string;
    detail?: string;
  }>;
}

interface PlayerBreakdown {
  player_id: string;
  shots_analyzed: number;
  dominant_shot_type: string;
  average_form_score: number;
  strengths: string[];
  areas_for_improvement: string[];
  shot_by_shot: ShotBreakdown[];
  trends: {
    form_trend?: string;
  };
  video_metrics: {
    total_frames: number;
    video_duration_seconds: number;
    analysis_timestamp: string;
  };
  average_metrics: {
    avg_layback: number;
    avg_h2ss: number;
    avg_hand_velocity: number;
    avg_trunk_rotation: number;
    avg_elbow_angle: number;
  };
}

interface AnalysisResult {
  rating: string;
  rating_color: string;
  score: number;
  comparison: string;
  main_feedback: string;
  shot_type: string;
  drills: Array<{
    name: string;
    description: string;
    coaching_point: string;
    reps: string;
  }>;
  feedback: Array<{
    type: string;
    message: string;
    detail?: string;
  }>;
  metrics: {
    avg_angle: number;
    max_angle: number;
    min_angle: number;
    angle_range: number;
    shots_detected: number;
    form_score: number;
    avg_load_angle: number;
    avg_release_angle: number;
    avg_explosion: number;
    arm_extension_style: string;
    arm_path_type: string;
    max_layback: number;
    max_h2ss: number;
    max_hand_velocity: number;
    trunk_rotation: number;
  };
  shots_analyzed: number;
  standards_used: string;
  person_detected?: boolean;
}

interface ApiResponse {
  analysis: AnalysisResult;
  player_breakdown: PlayerBreakdown;
}

export default function UploadPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [processedVideoUrl, setProcessedVideoUrl] = useState<string>('');
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult | null>(null);
  const [playerBreakdown, setPlayerBreakdown] = useState<PlayerBreakdown | null>(null);
  const [uploadProgress, setUploadProgress] = useState<{[key: string]: number}>({});
  const [uploadStatus, setUploadStatus] = useState<{[key: string]: 'pending' | 'uploading' | 'success' | 'error'}>({});
  const [error, setError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [activeTab, setActiveTab] = useState<'analysis' | 'breakdown'>('analysis');
  const [expandedShots, setExpandedShots] = useState<number[]>([]);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Check backend status on mount and periodically
  useEffect(() => {
    checkBackendStatus();
    const interval = setInterval(checkBackendStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const checkBackendStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/test', {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
        mode: 'cors',
      });
      
      if (response.ok) {
        setBackendStatus('online');
        setError(null);
      } else {
        setBackendStatus('offline');
      }
    } catch (err) {
      console.log('Backend connection failed:', err);
      setBackendStatus('offline');
    }
  };

  // Clean up object URLs
  useEffect(() => {
    return () => {
      if (processedVideoUrl) {
        URL.revokeObjectURL(processedVideoUrl);
      }
    };
  }, [processedVideoUrl]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const videoFiles = acceptedFiles.filter(file => 
      file.type.startsWith('video/')
    );
    
    setFiles(prev => [...prev, ...videoFiles]);
    setError(null);
    setAnalysisResults(null);
    setPlayerBreakdown(null);
    if (processedVideoUrl) {
      URL.revokeObjectURL(processedVideoUrl);
      setProcessedVideoUrl('');
    }
  }, [processedVideoUrl]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    },
    maxSize: 104857600, // 100MB
    disabled: backendStatus === 'offline'
  });

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getRatingColorClass = (color: string) => {
    const colorMap: {[key: string]: string} = {
      'purple': 'bg-purple-100 text-purple-800 border-purple-200',
      'blue': 'bg-blue-100 text-blue-800 border-blue-200',
      'green': 'bg-green-100 text-green-800 border-green-200',
      'yellow': 'bg-yellow-100 text-yellow-800 border-yellow-200',
      'orange': 'bg-orange-100 text-orange-800 border-orange-200',
      'red': 'bg-red-100 text-red-800 border-red-200',
      'gray': 'bg-gray-100 text-gray-800 border-gray-200'
    };
    return colorMap[color] || 'bg-gray-100 text-gray-800 border-gray-200';
  };

  const toggleShotExpanded = (shotNumber: number) => {
    setExpandedShots(prev => 
      prev.includes(shotNumber) 
        ? prev.filter(n => n !== shotNumber)
        : [...prev, shotNumber]
    );
  };

  const uploadAndAnalyze = async (file: File) => {
    const fileId = file.name;
    
    setUploadStatus(prev => ({ ...prev, [fileId]: 'uploading' }));

    const formData = new FormData();
    formData.append('video', file);

    try {
      // Simulate upload progress
      for (let progress = 0; progress <= 100; progress += 10) {
        await new Promise(resolve => setTimeout(resolve, 100));
        setUploadProgress(prev => ({ ...prev, [fileId]: progress }));
      }

      console.log('Attempting to connect to backend...');
      
      // First, test the connection
      try {
        const testResponse = await fetch('http://localhost:8000/api/test', {
          method: 'GET',
          headers: { 'Accept': 'application/json' },
        });
        console.log('Backend test response:', testResponse.status);
      } catch (testError) {
        console.error('Backend not reachable:', testError);
        throw new Error('Cannot connect to backend server. Please make sure it\'s running on port 8000');
      }

      // Upload to backend with timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000);

      console.log('Sending video to backend...');
      
      const response = await fetch('http://localhost:8000/api/analyze-video', {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      }).finally(() => clearTimeout(timeoutId));

      console.log('Response received:', response.status);

      if (!response.ok) {
        let errorMessage = 'Upload failed';
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorMessage;
        } catch {
          errorMessage = await response.text() || errorMessage;
        }
        throw new Error(errorMessage);
      }

      // Check if the response is a video
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('video/')) {
        throw new Error('Server did not return a video file');
      }

      // Get analysis results from headers
      const analysisHeader = response.headers.get('X-Analysis-Results');
      if (analysisHeader) {
        try {
          const apiResponse: ApiResponse = JSON.parse(analysisHeader);
          setAnalysisResults(apiResponse.analysis);
          setPlayerBreakdown(apiResponse.player_breakdown);
          console.log('Analysis results:', apiResponse.analysis);
          console.log('Player breakdown:', apiResponse.player_breakdown);
        } catch (e) {
          console.error('Failed to parse analysis results:', e);
        }
      }

      // Get the video blob from the response
      const videoBlob = await response.blob();
      
      // Create URL for the video blob
      const videoUrl = URL.createObjectURL(videoBlob);
      
      setProcessedVideoUrl(videoUrl);
      setUploadStatus(prev => ({ ...prev, [fileId]: 'success' }));
      
      return videoUrl;
    } catch (error) {
      setUploadStatus(prev => ({ ...prev, [fileId]: 'error' }));
      
      let errorMessage = 'Upload failed';
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorMessage = 'Upload timed out. The video might be too large or the server is slow.';
        } else {
          errorMessage = error.message;
        }
      }
      
      setError(errorMessage);
      console.error('Upload error:', error);
      throw error;
    }
  };

  const handleUpload = async () => {
    if (files.length === 0) return;
    
    setUploading(true);
    setError(null);
    
    try {
      await uploadAndAnalyze(files[0]);
    } catch (error) {
      // Error already handled
    } finally {
      setUploading(false);
    }
  };

  const handleUploadAnother = () => {
    if (processedVideoUrl) {
      URL.revokeObjectURL(processedVideoUrl);
    }
    
    setFiles([]);
    setProcessedVideoUrl('');
    setAnalysisResults(null);
    setPlayerBreakdown(null);
    setUploadStatus({});
    setUploadProgress({});
    setError(null);
    setActiveTab('analysis');
    setExpandedShots([]);
  };

  const handleVideoError = (e: React.SyntheticEvent<HTMLVideoElement, Event>) => {
    console.error('Video playback error:', e);
    setError('Error playing back the analyzed video. The file may be corrupted.');
  };

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Upload & Analyze Lacrosse Shots
          </h1>
          <p className="text-gray-600">
            Upload your lacrosse shot video for AI-powered form analysis
          </p>
          
          {/* Backend Status Indicator */}
          <div className="mt-4 flex items-center justify-center space-x-2">
            <div className={`h-3 w-3 rounded-full ${
              backendStatus === 'online' ? 'bg-green-500' : 
              backendStatus === 'offline' ? 'bg-red-500' : 
              'bg-yellow-500 animate-pulse'
            }`} />
            <span className="text-sm text-gray-600">
              {backendStatus === 'online' ? 'Backend connected' : 
               backendStatus === 'offline' ? 'Backend offline' : 
               'Checking backend...'}
            </span>
            {backendStatus === 'offline' && (
              <button
                onClick={checkBackendStatus}
                className="text-sm text-blue-600 hover:underline ml-2"
              >
                Retry
              </button>
            )}
          </div>
        </div>

        {/* Backend Offline Message */}
        {backendStatus === 'offline' && (
          <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <p className="text-yellow-800">
              ⚠️ Backend server is not running. Please start it with:
            </p>
            <pre className="mt-2 p-2 bg-gray-800 text-white rounded text-sm">
              cd backend &amp;&amp; source venv/bin/activate &amp;&amp; python -m uvicorn main:app --reload --port 8000
            </pre>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center space-x-2">
            <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0" />
            <p className="text-red-700">{error}</p>
          </div>
        )}

        {/* Upload Area */}
        {!processedVideoUrl && backendStatus === 'online' && (
          <div
            {...getRootProps()}
            className={`
              border-2 border-dashed rounded-lg p-12 text-center cursor-pointer
              transition-colors duration-200 mb-8
              ${isDragActive 
                ? 'border-blue-500 bg-blue-50' 
                : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50'
              }
            `}
          >
            <input {...getInputProps()} />
            <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            {isDragActive ? (
              <p className="text-blue-600">Drop your videos here...</p>
            ) : (
              <div>
                <p className="text-gray-700 mb-2">
                  Drag & drop videos here, or click to select
                </p>
                <p className="text-sm text-gray-500">
                  Supports: MP4, MOV, AVI, MKV, WEBM (Max 100MB)
                </p>
              </div>
            )}
          </div>
        )}

        {/* File List */}
        {files.length > 0 && !processedVideoUrl && backendStatus === 'online' && (
          <div className="mt-8">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Selected Files ({files.length})
            </h2>
            <div className="space-y-3">
              {files.map((file, index) => {
                const fileId = file.name;
                const status = uploadStatus[fileId] || 'pending';
                const progress = uploadProgress[fileId] || 0;

                return (
                  <div
                    key={`${file.name}-${index}`}
                    className="bg-white rounded-lg p-4 shadow-sm border border-gray-200"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-3">
                        <Film className="h-5 w-5 text-gray-400" />
                        <div>
                          <p className="text-sm font-medium text-gray-900">
                            {file.name}
                          </p>
                          <p className="text-xs text-gray-500">
                            {formatFileSize(file.size)}
                          </p>
                        </div>
                      </div>
                      {status === 'pending' && !uploading && (
                        <button
                          onClick={() => removeFile(index)}
                          className="text-gray-400 hover:text-gray-600"
                        >
                          <X className="h-5 w-5" />
                        </button>
                      )}
                      {status === 'success' && (
                        <CheckCircle className="h-5 w-5 text-green-500" />
                      )}
                      {status === 'error' && (
                        <AlertCircle className="h-5 w-5 text-red-500" />
                      )}
                    </div>

                    {/* Progress Bar */}
                    {status === 'uploading' && (
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${progress}%` }}
                        />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Upload Button */}
            <div className="mt-6">
              <button
                onClick={handleUpload}
                disabled={uploading || files.length === 0}
                className={`
                  w-full py-3 px-4 rounded-lg font-medium
                  transition-colors duration-200 flex items-center justify-center space-x-2
                  ${uploading || files.length === 0
                    ? 'bg-gray-300 cursor-not-allowed text-gray-500'
                    : 'bg-blue-600 hover:bg-blue-700 text-white'
                  }
                `}
              >
                {uploading ? (
                  <>
                    <Loader2 className="h-5 w-5 animate-spin" />
                    <span>Analyzing...</span>
                  </>
                ) : (
                  <span>Analyze {files.length} Video{files.length !== 1 ? 's' : ''}</span>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Results Section */}
        {processedVideoUrl && analysisResults && (
          <div className="mt-8 space-y-6">
            {/* Video Player */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">Analyzed Video</h2>
              <div className="relative bg-black rounded-lg overflow-hidden">
                <video 
                  ref={videoRef}
                  controls 
                  className="w-full rounded-lg shadow-md"
                  src={processedVideoUrl}
                  onError={handleVideoError}
                  onLoadedData={() => console.log('Video loaded successfully')}
                >
                  Your browser does not support the video tag.
                </video>
              </div>
            </div>

            {/* Tab Navigation */}
            <div className="flex border-b border-gray-200">
              <button
                className={`py-2 px-4 font-medium text-sm flex items-center space-x-2 ${
                  activeTab === 'analysis'
                    ? 'text-blue-600 border-b-2 border-blue-600'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
                onClick={() => setActiveTab('analysis')}
              >
                <BarChart className="h-4 w-4" />
                <span>Shot Analysis</span>
              </button>
              <button
                className={`py-2 px-4 font-medium text-sm flex items-center space-x-2 ${
                  activeTab === 'breakdown'
                    ? 'text-blue-600 border-b-2 border-blue-600'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
                onClick={() => setActiveTab('breakdown')}
              >
                <User className="h-4 w-4" />
                <span>Player Breakdown</span>
              </button>
            </div>

            {/* Analysis Tab */}
            {activeTab === 'analysis' && (
              <div className="space-y-6">
                {/* Rating Card */}
                <div className={`rounded-lg border-2 p-6 ${getRatingColorClass(analysisResults.rating_color)}`}>
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <p className="text-sm font-medium opacity-75">Form Rating</p>
                      <h2 className="text-3xl font-bold">{analysisResults.rating}</h2>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-medium opacity-75">Score</p>
                      <p className="text-4xl font-bold">{Math.round(analysisResults.score)}</p>
                    </div>
                  </div>
                  <p className="text-lg font-medium">{analysisResults.comparison}</p>
                </div>

                {/* Key Metrics Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-200">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-gray-500">Shots</span>
                      <Target className="h-4 w-4 text-blue-500" />
                    </div>
                    <p className="text-2xl font-bold text-gray-900">{analysisResults.metrics.shots_detected}</p>
                  </div>
                  
                  <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-200">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-gray-500">Avg Angle</span>
                      <TrendingUp className="h-4 w-4 text-green-500" />
                    </div>
                    <p className="text-2xl font-bold text-gray-900">{Math.round(analysisResults.metrics.avg_angle)}°</p>
                  </div>
                  
                  <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-200">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-gray-500">Release</span>
                      <Zap className="h-4 w-4 text-yellow-500" />
                    </div>
                    <p className="text-2xl font-bold text-gray-900">{Math.round(analysisResults.metrics.avg_release_angle)}°</p>
                  </div>
                  
                  <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-200">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-gray-500">Explosion</span>
                      <Award className="h-4 w-4 text-purple-500" />
                    </div>
                    <p className="text-2xl font-bold text-gray-900">{Math.round(analysisResults.metrics.avg_explosion)}°</p>
                  </div>
                </div>

                {/* Advanced Metrics */}
                <div className="bg-white rounded-lg shadow-lg p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Advanced Metrics</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                      <p className="text-sm text-gray-500">Max Layback</p>
                      <p className="text-xl font-bold text-gray-900">{Math.round(analysisResults.metrics.max_layback)}°</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Max H2SS</p>
                      <p className="text-xl font-bold text-gray-900">{Math.round(analysisResults.metrics.max_h2ss)}°</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Hand Velocity</p>
                      <p className="text-xl font-bold text-gray-900">{analysisResults.metrics.max_hand_velocity.toFixed(1)}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Trunk Rotation</p>
                      <p className="text-xl font-bold text-gray-900">{Math.round(analysisResults.metrics.trunk_rotation)}°</p>
                    </div>
                  </div>
                  <div className="mt-4 grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-gray-500">Arm Style</p>
                      <p className="font-medium text-gray-900 capitalize">{analysisResults.metrics.arm_extension_style.replace('_', ' ')}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Path Type</p>
                      <p className="font-medium text-gray-900 capitalize">{analysisResults.metrics.arm_path_type.replace('_', ' ')}</p>
                    </div>
                  </div>
                </div>

                {/* Main Feedback */}
                <div className="bg-blue-50 rounded-lg p-6 border border-blue-200">
                  <p className="text-lg text-blue-900">{analysisResults.main_feedback}</p>
                </div>

                {/* Feedback List */}
                {analysisResults.feedback && analysisResults.feedback.length > 0 && (
                  <div className="bg-white rounded-lg shadow-lg p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Detailed Feedback</h3>
                    <ul className="space-y-3">
                      {analysisResults.feedback.map((item, index) => (
                        <li key={index} className="flex items-start space-x-2">
                          {item.type === 'excellent' && <CheckCircle className="h-5 w-5 text-green-500 flex-shrink-0 mt-0.5" />}
                          {item.type === 'good' && <CheckCircle className="h-5 w-5 text-blue-500 flex-shrink-0 mt-0.5" />}
                          {item.type === 'moderate' && <AlertCircle className="h-5 w-5 text-yellow-500 flex-shrink-0 mt-0.5" />}
                          {item.type === 'needs_work' && <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />}
                          {item.type === 'tip' && <Zap className="h-5 w-5 text-purple-500 flex-shrink-0 mt-0.5" />}
                          {item.type === 'metric' && <TrendingUp className="h-5 w-5 text-gray-500 flex-shrink-0 mt-0.5" />}
                          {item.type === 'insight' && <Award className="h-5 w-5 text-orange-500 flex-shrink-0 mt-0.5" />}
                          <div>
                            <p className="text-gray-700">{item.message}</p>
                            {item.detail && <p className="text-sm text-gray-500 mt-1">{item.detail}</p>}
                          </div>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Drills */}
                {analysisResults.drills && analysisResults.drills.length > 0 && (
                  <div className="bg-white rounded-lg shadow-lg p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Recommended Drills</h3>
                    <div className="space-y-4">
                      {analysisResults.drills.map((drill, index) => (
                        <div key={index} className="border-l-4 border-blue-500 pl-4 py-2">
                          <h4 className="font-semibold text-gray-900">{drill.name}</h4>
                          <p className="text-sm text-gray-600 mt-1">{drill.description}</p>
                          <p className="text-sm text-blue-600 mt-1 italic">💡 {drill.coaching_point}</p>
                          <p className="text-xs text-gray-500 mt-2">{drill.reps}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Player Breakdown Tab */}
            {activeTab === 'breakdown' && playerBreakdown && (
              <div className="space-y-6">
                {/* Player Summary */}
                <div className="bg-white rounded-lg shadow-lg p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-bold text-gray-900">Player Profile</h3>
                    <span className="text-sm text-gray-500">
                      {playerBreakdown.shots_analyzed} shot{playerBreakdown.shots_analyzed !== 1 ? 's' : ''} analyzed
                    </span>
                  </div>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    <div>
                      <p className="text-sm text-gray-500">Avg Form Score</p>
                      <p className="text-2xl font-bold text-gray-900">{Math.round(playerBreakdown.average_form_score)}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Dominant Shot</p>
                      <p className="text-2xl font-bold text-gray-900 capitalize">{playerBreakdown.dominant_shot_type}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Trend</p>
                      <p className="text-2xl font-bold text-gray-900 capitalize">{playerBreakdown.trends.form_trend || 'N/A'}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Duration</p>
                      <p className="text-2xl font-bold text-gray-900">{playerBreakdown.video_metrics.video_duration_seconds.toFixed(1)}s</p>
                    </div>
                  </div>

                  {/* Strengths */}
                  {playerBreakdown.strengths.length > 0 && (
                    <div className="mb-4">
                      <h4 className="font-semibold text-green-700 mb-2">💪 Strengths</h4>
                      <ul className="space-y-1">
                        {playerBreakdown.strengths.map((strength, i) => (
                          <li key={i} className="flex items-start space-x-2 text-gray-700">
                            <CheckCircle className="h-4 w-4 text-green-500 flex-shrink-0 mt-0.5" />
                            <span>{strength}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Areas for Improvement */}
                  {playerBreakdown.areas_for_improvement.length > 0 && (
                    <div>
                      <h4 className="font-semibold text-orange-700 mb-2">🎯 Areas to Improve</h4>
                      <ul className="space-y-1">
                        {playerBreakdown.areas_for_improvement.map((area, i) => (
                          <li key={i} className="flex items-start space-x-2 text-gray-700">
                            <AlertCircle className="h-4 w-4 text-orange-500 flex-shrink-0 mt-0.5" />
                            <span>{area}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>

                {/* Average Metrics */}
                <div className="bg-white rounded-lg shadow-lg p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Average Metrics</h3>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                    <div>
                      <p className="text-sm text-gray-500">Layback</p>
                      <p className="text-xl font-bold text-gray-900">{Math.round(playerBreakdown.average_metrics.avg_layback)}°</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">H2SS</p>
                      <p className="text-xl font-bold text-gray-900">{Math.round(playerBreakdown.average_metrics.avg_h2ss)}°</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Hand Speed</p>
                      <p className="text-xl font-bold text-gray-900">{playerBreakdown.average_metrics.avg_hand_velocity.toFixed(1)}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Trunk</p>
                      <p className="text-xl font-bold text-gray-900">{Math.round(playerBreakdown.average_metrics.avg_trunk_rotation)}°</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Elbow</p>
                      <p className="text-xl font-bold text-gray-900">{Math.round(playerBreakdown.average_metrics.avg_elbow_angle)}°</p>
                    </div>
                  </div>
                </div>

                {/* Shot-by-Shot Breakdown */}
                <div className="bg-white rounded-lg shadow-lg p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Shot-by-Shot Breakdown</h3>
                  <div className="space-y-4">
                    {playerBreakdown.shot_by_shot.map((shot) => (
                      <div key={shot.shot_number} className="border border-gray-200 rounded-lg overflow-hidden">
                        <button
                          onClick={() => toggleShotExpanded(shot.shot_number)}
                          className="w-full px-4 py-3 bg-gray-50 hover:bg-gray-100 flex items-center justify-between"
                        >
                          <div className="flex items-center space-x-4">
                            <span className="font-semibold text-gray-900">Shot #{shot.shot_number}</span>
                            <span className="text-sm text-gray-600">at {shot.timestamp.toFixed(1)}s</span>
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                              shot.form_score >= 80 ? 'bg-green-100 text-green-800' :
                              shot.form_score >= 60 ? 'bg-yellow-100 text-yellow-800' :
                              'bg-red-100 text-red-800'
                            }`}>
                              Score: {Math.round(shot.form_score)}
                            </span>
                            <span className="text-sm text-gray-600 capitalize">{shot.shot_type}</span>
                          </div>
                          {expandedShots.includes(shot.shot_number) ? (
                            <ChevronUp className="h-5 w-5 text-gray-500" />
                          ) : (
                            <ChevronDown className="h-5 w-5 text-gray-500" />
                          )}
                        </button>
                        
                        {expandedShots.includes(shot.shot_number) && (
                          <div className="p-4 border-t border-gray-200">
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                              <div>
                                <p className="text-xs text-gray-500">Load Angle</p>
                                <p className="font-medium">{Math.round(shot.metrics.load_angle)}°</p>
                              </div>
                              <div>
                                <p className="text-xs text-gray-500">Release Angle</p>
                                <p className="font-medium">{Math.round(shot.metrics.release_angle)}°</p>
                              </div>
                              <div>
                                <p className="text-xs text-gray-500">Angle Change</p>
                                <p className="font-medium">{Math.round(shot.metrics.angle_change)}°</p>
                              </div>
                              <div>
                                <p className="text-xs text-gray-500">Shot Speed</p>
                                <p className="font-medium">{shot.metrics.shot_speed.toFixed(1)}</p>
                              </div>
                              <div>
                                <p className="text-xs text-gray-500">Layback</p>
                                <p className="font-medium">{Math.round(shot.metrics.max_layback)}°</p>
                              </div>
                              <div>
                                <p className="text-xs text-gray-500">H2SS</p>
                                <p className="font-medium">{Math.round(shot.metrics.max_h2ss)}°</p>
                              </div>
                              <div>
                                <p className="text-xs text-gray-500">Hand Velocity</p>
                                <p className="font-medium">{shot.metrics.max_hand_velocity.toFixed(1)}</p>
                              </div>
                              <div>
                                <p className="text-xs text-gray-500">Trunk Rotation</p>
                                <p className="font-medium">{Math.round(shot.metrics.trunk_rotation)}°</p>
                              </div>
                            </div>
                            
                            <div className="flex flex-wrap gap-2">
                              <span className="px-2 py-1 bg-gray-100 rounded-full text-xs text-gray-700 capitalize">
                                Arm: {shot.metrics.arm_extension_style.replace('_', ' ')}
                              </span>
                              <span className="px-2 py-1 bg-gray-100 rounded-full text-xs text-gray-700 capitalize">
                                Path: {shot.metrics.arm_path_type.replace('_', ' ')}
                              </span>
                            </div>

                            {shot.feedback && shot.feedback.length > 0 && (
                              <div className="mt-4 pt-4 border-t border-gray-200">
                                <p className="text-sm font-medium text-gray-700 mb-2">Shot Feedback:</p>
                                <ul className="space-y-1">
                                  {shot.feedback.map((fb, idx) => (
                                    <li key={idx} className="text-sm text-gray-600 flex items-start space-x-2">
                                      {fb.type === 'excellent' && <CheckCircle className="h-4 w-4 text-green-500 flex-shrink-0 mt-0.5" />}
                                      {fb.type === 'good' && <CheckCircle className="h-4 w-4 text-blue-500 flex-shrink-0 mt-0.5" />}
                                      {fb.type === 'needs_work' && <AlertCircle className="h-4 w-4 text-red-500 flex-shrink-0 mt-0.5" />}
                                      <span>{fb.message}</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Video Info */}
                <div className="bg-gray-50 rounded-lg p-4 text-sm text-gray-600">
                  <p>Analysis completed: {new Date(playerBreakdown.video_metrics.analysis_timestamp).toLocaleString()}</p>
                  <p>Frames processed: {playerBreakdown.video_metrics.total_frames}</p>
                </div>
              </div>
            )}

            {/* Upload Another Button */}
            <button
              onClick={handleUploadAnother}
              className="w-full py-3 px-4 bg-gray-200 hover:bg-gray-300 rounded-lg font-medium transition-colors"
            >
              Upload Another Video
            </button>
          </div>
        )}

        {/* Upload Guidelines */}
        {!processedVideoUrl && !files.length && backendStatus === 'online' && (
          <div className="mt-8 bg-blue-50 rounded-lg p-6">
            <h3 className="text-sm font-semibold text-blue-900 mb-3">
              Upload Guidelines
            </h3>
            <ul className="space-y-2 text-sm text-blue-700">
              <li>• Maximum file size: 100MB</li>
              <li>• Supported formats: MP4, MOV, AVI, MKV, WEBM</li>
              <li>• Best results with clear, well-lit videos of the full body</li>
              <li>• The AI analyzes elbow angle, forearm layback, hip-shoulder separation, and more</li>
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
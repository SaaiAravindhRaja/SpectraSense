import React, { useState, useRef, useCallback, useEffect } from 'react'

// Glassmorphism Upload Dropzone
function UploadDropzone({ onPick, disabled }) {
  const [over, setOver] = useState(false)
  const [error, setError] = useState(null)
  const ref = useRef()

  const validateFile = (file) => {
    const maxSize = 10 * 1024 * 1024 // 10MB
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/heic', 'image/webp']

    if (file.size > maxSize) {
      return 'File size must be less than 10MB'
    }

    if (!allowedTypes.includes(file.type) && !file.name.toLowerCase().match(/\.(jpg|jpeg|png|heic|webp)$/)) {
      return 'Please upload a valid image file (JPEG, PNG, HEIC, WebP)'
    }

    return null
  }

  const onDrop = useCallback((ev) => {
    ev.preventDefault()
    setOver(false)
    setError(null)

    const file = ev.dataTransfer.files && ev.dataTransfer.files[0]
    if (file) {
      const validationError = validateFile(file)
      if (validationError) {
        setError(validationError)
        return
      }
      onPick(file)
    }
  }, [onPick])

  const onFileSelect = (e) => {
    setError(null)
    const file = e.target.files[0]
    if (file) {
      const validationError = validateFile(file)
      if (validationError) {
        setError(validationError)
        return
      }
      onPick(file)
    }
  }

  return (
    <div className="space-y-4">
      <div ref={ref}
        onDragOver={(e) => { e.preventDefault(); if (!disabled) setOver(true) }}
        onDragLeave={() => setOver(false)}
        onDrop={disabled ? undefined : onDrop}
        className={`glass-card border-2 ${disabled ? 'border-white/20 cursor-not-allowed opacity-60' :
            over ? 'border-cyan-400/60 bg-cyan-50/10 scale-[1.02]' : 'border-white/30 hover:border-white/50'
          } border-dashed rounded-2xl p-12 text-center transition-all duration-300 backdrop-blur-xl`}
      >
        <div className="mb-6">
          <div className={`mx-auto w-20 h-20 rounded-full flex items-center justify-center ${disabled ? 'bg-slate-400/20' : 'bg-gradient-to-br from-cyan-400/20 to-blue-500/20'
            } backdrop-blur-sm`}>
            <svg className={`h-10 w-10 ${disabled ? 'text-slate-400' : 'text-cyan-400'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" d="M7 16l5-5 5 5M12 4v12" />
            </svg>
          </div>
        </div>
        <div className={`text-xl font-semibold mb-2 ${disabled ? 'text-slate-400' : 'text-white'}`}>
          {disabled ? 'Analyzing Image...' : 'Drop your lip photo here'}
        </div>
        <div className={`text-sm mb-8 ${disabled ? 'text-slate-400' : 'text-white/70'}`}>
          or click to browse your files
        </div>
        <div className="space-y-4">
          <button
            onClick={() => !disabled && ref.current?.querySelector('input')?.click()}
            disabled={disabled}
            className={`px-8 py-4 rounded-xl font-semibold transition-all duration-300 ${disabled
                ? 'bg-slate-400/20 text-slate-400 cursor-not-allowed'
                : 'bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white shadow-lg hover:shadow-xl hover:scale-105'
              }`}
          >
            {disabled ? (
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 border-2 border-slate-400 border-t-transparent rounded-full animate-spin"></div>
                <span>Processing...</span>
              </div>
            ) : 'Choose File'}
          </button>
          <div className="text-xs text-white/50">
            Supported: JPEG, PNG, HEIC, WebP • Max 10MB
          </div>
        </div>
        <input
          type="file"
          accept="image/*"
          onChange={onFileSelect}
          className="hidden"
          disabled={disabled}
        />
      </div>
      {error && (
        <div className="glass-card border border-red-400/30 bg-red-500/10 p-4 rounded-xl text-red-300 text-sm backdrop-blur-xl">
          <div className="flex items-center space-x-2">
            <svg className="w-4 h-4 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
            <span>{error}</span>
          </div>
        </div>
      )}
    </div>
  )
}

// Glassmorphism Results Display
function ResultsPanel({ result, loading }) {
  if (loading) {
    return (
      <div className="glass-card p-8 rounded-2xl backdrop-blur-xl border border-white/20">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-gradient-to-br from-cyan-400/20 to-blue-500/20 flex items-center justify-center backdrop-blur-sm">
            <div className="w-8 h-8 border-3 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
          </div>
          <div className="space-y-3">
            <div className="h-4 bg-white/20 rounded-full w-3/4 mx-auto animate-pulse"></div>
            <div className="h-6 bg-white/20 rounded-full w-1/2 mx-auto animate-pulse"></div>
            <div className="h-3 bg-white/20 rounded-full w-2/3 mx-auto animate-pulse"></div>
          </div>
          <div className="text-white/70 mt-4">Analyzing your image...</div>
        </div>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="glass-card p-8 rounded-2xl backdrop-blur-xl border border-white/20 text-center">
        <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-gradient-to-br from-slate-400/20 to-slate-500/20 flex items-center justify-center backdrop-blur-sm">
          <svg className="h-8 w-8 text-white/50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
        </div>
        <div className="text-white/70 text-lg">Ready for Analysis</div>
        <div className="text-white/50 text-sm mt-2">Upload a clear lip photo to get started</div>
      </div>
    )
  }

  if (!result.ok) {
    return (
      <div className="glass-card p-6 rounded-2xl backdrop-blur-xl border border-red-400/30 bg-red-500/10">
        <div className="flex items-start space-x-3">
          <div className="w-10 h-10 rounded-full bg-red-500/20 flex items-center justify-center flex-shrink-0">
            <svg className="h-5 w-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
          </div>
          <div>
            <div className="font-semibold text-red-300 mb-1">Analysis Failed</div>
            <div className="text-red-200 text-sm">{result.data.error}</div>
          </div>
        </div>
      </div>
    )
  }

  const data = result.data
  const getStatusColor = (status) => {
    switch (status) {
      case 'low': return 'from-red-400 to-red-600'
      case 'borderline_low': return 'from-orange-400 to-orange-600'
      case 'normal': return 'from-green-400 to-green-600'
      case 'borderline_high': return 'from-orange-400 to-orange-600'
      case 'high': return 'from-red-400 to-red-600'
      default: return 'from-slate-400 to-slate-600'
    }
  }

  const getConfidenceColor = (confidence) => {
    if (confidence >= 85) return 'from-green-400 to-green-600'
    if (confidence >= 70) return 'from-yellow-400 to-yellow-600'
    return 'from-orange-400 to-orange-600'
  }

  return (
    <div className="space-y-6">
      {/* Main Result Card */}
      <div className="glass-card p-8 rounded-2xl backdrop-blur-xl border border-white/20">
        <div className="text-center">
          <div className="text-5xl font-bold text-white mb-4 tracking-tight">
            {data.prediction} <span className="text-2xl text-white/70">g/dL</span>
          </div>
          <div className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-semibold bg-gradient-to-r ${getStatusColor(data.interpretation.status)} text-white shadow-lg`}>
            {data.interpretation.status.replace('_', ' ').toUpperCase()}
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-white/10">
          <div className="flex justify-between items-center mb-3">
            <span className="text-white/70 font-medium">Confidence Score</span>
            <span className="text-white font-semibold text-lg">
              {data.confidence}%
            </span>
          </div>
          <div className="relative">
            <div className="w-full bg-white/10 rounded-full h-3 overflow-hidden">
              <div
                className={`h-full rounded-full bg-gradient-to-r ${getConfidenceColor(data.confidence)} transition-all duration-1000 ease-out shadow-lg`}
                style={{ width: `${data.confidence}%` }}
              ></div>
            </div>
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent rounded-full animate-pulse"></div>
          </div>
        </div>
      </div>

      {/* Interpretation Card */}
      <div className="glass-card p-6 rounded-2xl backdrop-blur-xl border border-white/20">
        <div className="flex items-start space-x-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-400/20 to-blue-500/20 flex items-center justify-center flex-shrink-0">
            <svg className="h-5 w-5 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div className="flex-1">
            <h4 className="font-semibold text-white mb-2">Medical Interpretation</h4>
            <p className="text-white/80 text-sm mb-3 leading-relaxed">{data.interpretation.message}</p>
            <div className="text-xs text-white/60 bg-white/5 rounded-lg p-3 border border-white/10">
              <strong className="text-white/80">Recommendation:</strong> {data.interpretation.recommendation}
            </div>
          </div>
        </div>
      </div>

      {/* Metadata Card */}
      {data.metadata && (
        <div className="glass-card p-4 rounded-xl backdrop-blur-xl border border-white/10">
          <div className="grid grid-cols-2 gap-4 text-xs">
            <div className="text-white/50">
              <div className="text-white/70 font-medium">Processing Time</div>
              <div>{data.metadata.processing_time_ms}ms</div>
            </div>
            <div className="text-white/50">
              <div className="text-white/70 font-medium">Model Type</div>
              <div>{data.metadata.model_type}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Glassmorphism Image Preview Component
function ImagePreview({ file, preview, onReset }) {
  return (
    <div className="glass-card p-6 rounded-2xl backdrop-blur-xl border border-white/20">
      <div className="flex items-start gap-6">
        <div className="relative group">
          <img
            src={preview}
            alt="preview"
            className="rounded-xl max-h-48 max-w-48 object-cover shadow-2xl border border-white/20 transition-transform duration-300 group-hover:scale-105"
          />
          <div className="absolute inset-0 rounded-xl bg-gradient-to-t from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
        </div>
        <div className="flex-1 min-w-0">
          <div className="font-semibold text-white text-lg truncate mb-2">{file?.name}</div>
          <div className="text-white/70 text-sm mb-4">
            {file && `${(file.size / 1024 / 1024).toFixed(1)} MB`} • Ready for analysis
          </div>
          <div className="glass-card p-4 rounded-xl border border-white/10 bg-white/5 mb-4">
            <div className="text-xs text-white/60 leading-relaxed">
              <div className="flex items-center space-x-2 mb-2">
                <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-white/80 font-medium">Image Quality Tips:</span>
              </div>
              <ul className="space-y-1 text-white/60">
                <li>• Lips clearly visible and well-lit</li>
                <li>• Centered in frame with good contrast</li>
                <li>• Natural lighting preferred</li>
              </ul>
            </div>
          </div>
          <button
            onClick={onReset}
            className="px-4 py-2 rounded-lg text-sm font-medium text-white/70 hover:text-white bg-white/10 hover:bg-white/20 border border-white/20 hover:border-white/30 transition-all duration-200"
          >
            Choose Different Image
          </button>
        </div>
      </div>
    </div>
  )
}

// Main App Component with Stunning Glassmorphism Design
function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [stats, setStats] = useState(null)

  // Load app stats on mount
  useEffect(() => {
    fetch('/api/stats')
      .then(res => res.json())
      .then(setStats)
      .catch(console.error)
  }, [])

  const onFile = (f) => {
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
  }

  const upload = async () => {
    if (!file) return
    setLoading(true)
    setResult(null)

    const fd = new FormData()
    fd.append('image', file)

    try {
      const res = await fetch('/predict', { method: 'POST', body: fd })
      const data = await res.json()
      setResult({ ok: res.ok, data })
    } catch (e) {
      setResult({ ok: false, data: { error: e.message } })
    }
    setLoading(false)
  }

  const reset = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    if (preview) URL.revokeObjectURL(preview)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0">
        <div className="absolute top-0 -left-4 w-72 h-72 bg-purple-300 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob"></div>
        <div className="absolute top-0 -right-4 w-72 h-72 bg-yellow-300 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-2000"></div>
        <div className="absolute -bottom-8 left-20 w-72 h-72 bg-pink-300 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-4000"></div>
        <div className="absolute bottom-0 right-20 w-72 h-72 bg-cyan-300 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-6000"></div>
      </div>

      {/* Glassmorphism Overlay */}
      <div className="absolute inset-0 bg-gradient-to-br from-white/10 via-white/5 to-transparent backdrop-blur-3xl"></div>

      <div className="relative z-10 container mx-auto px-6 py-12">
        {/* Stunning Header */}
        <header className="text-center mb-16">
          <div className="inline-flex items-center justify-center w-20 h-20 glass-card rounded-3xl mb-6 backdrop-blur-xl border border-white/20">
            <svg className="w-10 h-10 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 7.172V5L8 4z" />
            </svg>
          </div>
          <h1 className="text-6xl font-bold bg-gradient-to-r from-white via-cyan-200 to-white bg-clip-text text-transparent mb-4 tracking-tight">
            SpectraSense
          </h1>
          <p className="text-xl text-white/80 mb-2 font-light">AI-Powered Hemoglobin Analysis</p>
          <p className="text-white/60 mb-6">Revolutionary non-invasive biomarker estimation through advanced computer vision</p>

          <div className="inline-flex items-center px-4 py-2 glass-card rounded-full text-sm font-medium text-amber-300 backdrop-blur-xl border border-amber-400/30 bg-amber-500/10">
            <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
            Research Prototype - Not for Medical Diagnosis
          </div>
        </header>

        {/* Main Content */}
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-12">
            {/* Left Column - Upload & Preview */}
            <div className="space-y-8">
              {!file ? (
                <UploadDropzone onPick={onFile} disabled={loading} />
              ) : (
                <ImagePreview file={file} preview={preview} onReset={reset} />
              )}

              {file && (
                <div className="flex gap-4">
                  <button
                    onClick={upload}
                    disabled={loading || !file}
                    className="flex-1 px-8 py-4 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white rounded-xl font-semibold disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 shadow-lg hover:shadow-xl hover:scale-105 backdrop-blur-xl"
                  >
                    {loading ? (
                      <div className="flex items-center justify-center">
                        <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin mr-3"></div>
                        Analyzing Image...
                      </div>
                    ) : (
                      <div className="flex items-center justify-center">
                        <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                        Analyze Image
                      </div>
                    )}
                  </button>
                  <button
                    onClick={reset}
                    className="px-6 py-4 glass-card hover:bg-white/20 text-white rounded-xl font-medium transition-all duration-200 backdrop-blur-xl border border-white/20 hover:border-white/30"
                  >
                    Reset
                  </button>
                </div>
              )}
            </div>

            {/* Right Column - Results */}
            <div className="space-y-6">
              <div className="glass-card p-2 rounded-2xl backdrop-blur-xl border border-white/20">
                <div className="glass-card p-6 rounded-xl backdrop-blur-xl border border-white/10">
                  <div className="flex items-center space-x-3 mb-6">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-400/20 to-blue-500/20 flex items-center justify-center">
                      <svg className="w-4 h-4 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                    </div>
                    <h3 className="text-xl font-semibold text-white">Analysis Results</h3>
                  </div>
                  <ResultsPanel result={result} loading={loading} />
                </div>
              </div>
            </div>
          </div>

          {/* Feature Cards */}
          <div className="mt-20 grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                icon: (
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                ),
                title: "Lightning Fast",
                description: "Advanced AI processing delivers results in under 500ms with state-of-the-art accuracy.",
                gradient: "from-yellow-400 to-orange-500"
              },
              {
                icon: (
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                  </svg>
                ),
                title: "Medical Grade",
                description: "Ensemble learning with uncertainty quantification provides reliable confidence scoring.",
                gradient: "from-green-400 to-emerald-500"
              },
              {
                icon: (
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                  </svg>
                ),
                title: "Privacy Secure",
                description: "Complete local processing ensures your medical data never leaves your device.",
                gradient: "from-purple-400 to-pink-500"
              }
            ].map((feature, index) => (
              <div key={index} className="glass-card p-8 rounded-2xl backdrop-blur-xl border border-white/20 hover:border-white/30 transition-all duration-300 group hover:scale-105">
                <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${feature.gradient} p-3 mb-6 text-white group-hover:scale-110 transition-transform duration-300`}>
                  {feature.icon}
                </div>
                <h4 className="text-xl font-semibold text-white mb-3">{feature.title}</h4>
                <p className="text-white/70 leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>

          {/* Footer */}
          <footer className="mt-20 text-center">
            <div className="glass-card p-8 rounded-2xl backdrop-blur-xl border border-white/20 inline-block">
              <div className="flex items-center justify-center gap-8 mb-6 text-sm">
                <a href="/health" className="text-white/70 hover:text-white transition-colors">System Status</a>
                <a href="/api/stats" className="text-white/70 hover:text-white transition-colors">API Documentation</a>
                <a href="/model_card.md" className="text-white/70 hover:text-white transition-colors">Model Details</a>
              </div>
              <div className="text-white/60 text-sm space-y-1">
                <p className="font-semibold text-white">SpectraSense v2.0 • Genesis1v1 DSA Case Competition</p>
                <p>Advancing non-invasive healthcare through artificial intelligence</p>
              </div>
            </div>
          </footer>
        </div>
      </div>
    </div>
  )
}

export default App

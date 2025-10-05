import React, {useState, useRef, useCallback} from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { GlassCard, PrimaryButton, SecondaryButton, ResultCard, LoadingSkeleton } from './components'

function UploadDropzone({onPick}){
  const [over, setOver] = useState(false)
  const ref = useRef()

  const onDrop = useCallback((ev)=>{
    ev.preventDefault()
    setOver(false)
    const f = ev.dataTransfer.files && ev.dataTransfer.files[0]
    if(f) onPick(f)
  },[onPick])

  return (
    <div ref={ref}
      onDragOver={(e)=>{e.preventDefault(); setOver(true)}}
      onDragLeave={()=>setOver(false)}
      onDrop={onDrop}
      className={`glass-card border-2 ${over? 'border-accent-primary shadow-glow-cyan-lg': 'border-border-glass'} border-dashed p-8 text-center transition-all duration-200`}
    >
      <motion.div
        animate={{ scale: over ? 1.05 : 1 }}
        transition={{ duration: 0.2 }}
      >
        <div className="mb-4">
          <svg className="mx-auto h-16 w-16 text-accent-primary opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
          </svg>
        </div>
        <h3 className="text-xl font-semibold text-text-primary mb-2">Drop lip image here</h3>
        <p className="text-sm text-text-secondary mb-6">or click to browse (JPEG/PNG/HEIC)</p>
        <PrimaryButton onClick={()=>ref.current && ref.current.querySelector('input') && ref.current.querySelector('input').click()}>
          Choose File
        </PrimaryButton>
      </motion.div>
      <input type="file" accept="image/*" onChange={e=>onPick(e.target.files[0])} className="hidden" />
    </div>
  )
}

function App(){
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const inputRef = useRef()

  const onFile = (f) => {
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
  }

  const upload = async () => {
    if(!file) return
    setLoading(true)
    setResult(null)
    const fd = new FormData()
    fd.append('image', file)
    try{
      const res = await fetch('/predict', {method:'POST', body: fd})
      let data = null
      try {
        data = await res.json()
      } catch (parseErr) {
        try { const txt = await res.text(); data = {error: txt}; } catch (e) { data = {error: parseErr.message}; }
      }
      setResult({ok: res.ok, data})
    }catch(e){
      setResult({ok:false, data: {error: e.message}})
    }
    setLoading(false)
  }

  return (
    <div className="relative min-h-screen bg-bg-primary flex items-center justify-center p-6 overflow-hidden">
      {/* Animated clinical gradient blobs */}
      <div className="bg-blob blob-1 rounded-full w-[500px] h-[500px] bg-gradient-to-tr from-accent-primary/30 via-accent-secondary/20 to-accent-primary/10 left-[-150px] top-[-100px]"></div>
      <div className="bg-blob blob-2 rounded-full w-[400px] h-[400px] bg-gradient-to-br from-accent-secondary/20 via-accent-primary/15 to-accent-secondary/10 right-[-120px] top-[-80px]"></div>
      <div className="bg-blob blob-3 rounded-full w-[350px] h-[350px] bg-gradient-to-bl from-accent-primary/15 via-accent-secondary/10 to-accent-primary/5 left-[15%] bottom-[-100px]"></div>

      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-5xl relative z-10"
      >
        <GlassCard className="p-10">
          {/* Header */}
          <motion.header
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="mb-8 text-center"
          >
            <h1 className="text-4xl font-bold text-text-primary mb-2 tracking-tight">
              SpectraSense<span className="text-accent-primary">.</span>
            </h1>
            <p className="text-text-secondary">Clinical-grade hemoglobin estimation from lip imagery</p>
          </motion.header>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Upload Section */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="lg:col-span-2"
            >
              <UploadDropzone onPick={onFile} />

              <AnimatePresence>
                {preview && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="mt-6"
                  >
                    <GlassCard className="p-4">
                      <div className="flex items-start gap-4">
                        <img src={preview} alt="preview" className="rounded-lg max-h-48 object-cover border border-border-glass" />
                        <div className="flex-1 min-w-0">
                          <h4 className="text-sm font-semibold text-text-primary mb-1 truncate">{file?.name}</h4>
                          <p className="caption mb-4">Ensure lips are centered, evenly lit, and occupy most of the frame.</p>
                          <div className="flex gap-3">
                            <PrimaryButton onClick={upload} disabled={loading || !file} loading={loading}>
                              {loading ? 'Analyzing' : 'Analyze'}
                            </PrimaryButton>
                            <SecondaryButton onClick={()=>{setFile(null); setPreview(null); setResult(null); inputRef.current && (inputRef.current.value='')}}>
                              Reset
                            </SecondaryButton>
                          </div>
                        </div>
                      </div>
                    </GlassCard>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* Results Section */}
            <motion.aside
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
            >
              <h3 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-4">Analysis Result</h3>
              
              <AnimatePresence mode="wait">
                {loading ? (
                  <motion.div
                    key="loading"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <LoadingSkeleton />
                  </motion.div>
                ) : result ? (
                  <motion.div
                    key="result"
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.9 }}
                  >
                    {result.ok ? (
                      (() => {
                        try {
                          const pred = Number(result.data.prediction)
                          if (!Number.isFinite(pred)) throw new Error('Invalid prediction')
                          
                          let status = 'neutral'
                          if (pred >= 12 && pred <= 16) status = 'success'
                          else if (pred < 10 || pred > 17) status = 'warning'
                          
                          return (
                            <ResultCard
                              title="Hemoglobin Level"
                              value={pred.toFixed(2)}
                              unit="g/dL"
                              status={status}
                              icon={
                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                              }
                            />
                          )
                        } catch (e) {
                          return (
                            <GlassCard className="p-4 border-error/30">
                              <p className="text-sm text-error">Unexpected output: {String(result.data.prediction)}</p>
                            </GlassCard>
                          )
                        }
                      })()
                    ) : (
                      <GlassCard className="p-4 border-error/30">
                        <div className="flex items-start gap-2">
                          <svg className="w-5 h-5 text-error flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          <div>
                            <p className="text-sm font-semibold text-error mb-1">Analysis Failed</p>
                            <p className="text-xs text-text-secondary">{result.data.error || 'Unknown error'}</p>
                          </div>
                        </div>
                      </GlassCard>
                    )}
                  </motion.div>
                ) : (
                  <motion.div
                    key="empty"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <GlassCard className="p-6 text-center">
                      <svg className="w-12 h-12 mx-auto text-text-secondary/40 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                      <p className="caption">Upload an image to begin analysis</p>
                    </GlassCard>
                  </motion.div>
                )}
              </AnimatePresence>

              <p className="caption mt-4 text-center">
                For HEIC images, convert to JPEG/PNG if upload fails.
              </p>
            </motion.aside>
          </div>

          {/* Footer */}
          <motion.footer
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
            className="mt-8 pt-6 border-t border-border-glass text-center"
          >
            <p className="caption">
              Research purposes only
            </p>
          </motion.footer>
        </GlassCard>
      </motion.div>
    </div>
  )
}

export default App
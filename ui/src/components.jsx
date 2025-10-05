import React from 'react'
import { motion } from 'framer-motion'

export function GlassCard({ children, className = '', hover = true, ...props }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`glass-card ${hover ? 'glow-border-hover' : ''} ${className}`}
      {...props}
    >
      {children}
    </motion.div>
  )
}

export function PrimaryButton({ children, onClick, disabled = false, loading = false, className = '', ...props }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled || loading}
      className={`btn-gradient px-6 py-3 rounded-lg font-semibold text-white disabled:opacity-50 disabled:cursor-not-allowed ${className}`}
      {...props}
    >
      {loading ? (
        <span className="flex items-center gap-2">
          <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          {children}
        </span>
      ) : children}
    </button>
  )
}

export function SecondaryButton({ children, onClick, disabled = false, className = '', ...props }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`px-6 py-3 rounded-lg font-semibold border border-border-glass text-text-primary hover:border-accent-primary hover:shadow-glow-cyan transition-all duration-150 disabled:opacity-50 disabled:cursor-not-allowed ${className}`}
      {...props}
    >
      {children}
    </button>
  )
}

export function ResultCard({ title, value, unit, icon, status = 'neutral', className = '' }) {
  const statusColors = {
    success: 'text-success border-success/30 shadow-[0_0_20px_rgba(74,222,128,0.2)]',
    warning: 'text-warning border-warning/30 shadow-[0_0_20px_rgba(250,204,21,0.2)]',
    error: 'text-error border-error/30 shadow-[0_0_20px_rgba(248,113,113,0.2)]',
    neutral: 'text-text-primary border-accent-primary/20'
  }

  return (
    <GlassCard className={`p-6 ${statusColors[status]} ${className}`}>
      <div className="flex items-start justify-between mb-4">
        <h3 className="text-sm font-medium text-text-secondary uppercase tracking-wider">{title}</h3>
        {icon && <div className="text-accent-primary opacity-60">{icon}</div>}
      </div>
      <div className="flex items-baseline gap-2">
        <span className="text-4xl font-bold">{value}</span>
        {unit && <span className="text-lg text-text-secondary">{unit}</span>}
      </div>
    </GlassCard>
  )
}

export function LoadingSkeleton({ className = '' }) {
  return (
    <div className={`shimmer rounded-lg bg-bg-glass h-20 ${className}`}></div>
  )
}

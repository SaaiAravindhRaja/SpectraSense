module.exports = {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'bg-primary': '#0E1117',
        'bg-glass': 'rgba(255,255,255,0.08)',
        'accent-primary': '#00B8D9',
        'accent-secondary': '#36CFC9',
        'text-primary': '#E6E8EB',
        'text-secondary': '#A0A6B0',
        'success': '#4ADE80',
        'warning': '#FACC15',
        'error': '#F87171',
        'border-glass': 'rgba(255,255,255,0.08)',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif'],
      },
      fontSize: {
        'h1': '2.5rem',
        'h2': '1.75rem',
        'caption': '0.85rem',
      },
      backdropBlur: {
        'clinical': '12px',
      },
      boxShadow: {
        'glass': '0 8px 30px rgba(0,0,0,0.2)',
        'glow-cyan': '0 0 20px rgba(0,184,217,0.3)',
        'glow-cyan-lg': '0 0 40px rgba(0,184,217,0.4)',
      },
      animation: {
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
        'shimmer': 'shimmer 2s linear infinite',
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { opacity: '1', boxShadow: '0 0 20px rgba(0,184,217,0.3)' },
          '50%': { opacity: '0.8', boxShadow: '0 0 40px rgba(0,184,217,0.5)' },
        },
        'shimmer': {
          '0%': { backgroundPosition: '-1000px 0' },
          '100%': { backgroundPosition: '1000px 0' },
        },
      },
    },
  },
  plugins: [],
}

import { useState, useEffect } from 'react'

interface CrspSummaryResponse {
  status: string;
  database: string;
  collection: string;
  total_count: number;
  sample_records: Record<string, any>[];
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:30080';

function App() {
  const [crspData, setCrspData] = useState<CrspSummaryResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'samples' | 'analytics'>('overview');

  const fetchAnalytics = () => {
    setLoading(true);
    setError(null);
    
    fetch(`${API_BASE_URL}/api/v1/analytics/crsp-summary`)
      .then((res) => {
        if (!res.ok) throw new Error('CRSP summary API response was not ok');
        return res.json();
      })
      .then((data) => {
        setCrspData(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchAnalytics();
  }, []);

  return (
    <div style={{
      minHeight: '100vh',
      backgroundColor: '#040511',
      color: '#0d161f',
      fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '2rem'
    }}>
      <div style={{
        width: '100%',
        maxWidth: '950px',
        background: '#1e293b',
        border: '1px solid #334155',
        borderRadius: '16px',
        padding: '2rem',
        boxShadow: '0 20px 25px -5px rgb(0 0 0 / 0.3), 0 8px 10px -6px rgb(0 0 0 / 0.3)'
      }}>
        {/* Header & Refresh */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem' }}>
          <div>
            <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700, letterSpacing: '-0.025em', color: '#f1f5f9' }}>
              NKEPSX Scientific Analytics
            </h1>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.875rem', color: '#94a3b8' }}>
              Quantitative Research & Database Telemetry Explorer
            </p>
          </div>
          <button
            onClick={fetchAnalytics}
            style={{
              background: '#3b82f6',
              color: '#ffffff',
              border: 'none',
              borderRadius: '8px',
              padding: '0.5rem 1rem',
              fontSize: '0.875rem',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'background 0.2s'
            }}
            onMouseOver={(e) => e.currentTarget.style.background = '#2563eb'}
            onMouseOut={(e) => e.currentTarget.style.background = '#3b82f6'}
          >
            Refresh Data
          </button>
        </div>

        {/* Navigation Tabs */}
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem', borderBottom: '1px solid #334155', paddingBottom: '0.75rem' }}>
          {(['overview', 'samples', 'analytics'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                background: activeTab === tab ? '#334155' : 'transparent',
                color: activeTab === tab ? '#f8fafc' : '#94a3b8',
                border: 'none',
                borderRadius: '6px',
                padding: '0.5rem 1rem',
                fontSize: '0.875rem',
                fontWeight: 600,
                cursor: 'pointer',
                textTransform: 'capitalize'
              }}
            >
              {tab === 'overview' ? 'Database Overview' : tab === 'samples' ? 'Sample Records' : 'Statistical Metrics'}
            </button>
          ))}
        </div>

        {/* Main Content Pane */}
        <div style={{
          background: '#0f172a',
          border: '1px solid #334155',
          borderRadius: '12px',
          padding: '1.25rem'
        }}>
          {loading && !crspData && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#94a3b8', padding: '1rem 0' }}>
              <span>⚡</span> Querying CRSP dataset metrics...
            </div>
          )}

          {error && (
            <div style={{ color: '#f87171', background: 'rgba(239, 68, 68, 0.1)', padding: '0.75rem', borderRadius: '8px', border: '1px solid rgba(239, 68, 68, 0.2)', marginBottom: '1rem' }}>
              <strong>Connection Error:</strong> {error}
            </div>
          )}

          {activeTab === 'overview' && crspData && (
            <div>
              <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', fontWeight: 600, color: '#cbd5e1' }}>
                MongoDB `insTrader` Core Metrics
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
                <div style={{ background: '#1e293b', padding: '1rem', borderRadius: '8px', border: '1px solid #334155' }}>
                  <span style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'block' }}>Target Database</span>
                  <strong style={{ color: '#e2e8f0', fontSize: '1rem' }}>{crspData.database}</strong>
                </div>
                <div style={{ background: '#1e293b', padding: '1rem', borderRadius: '8px', border: '1px solid #334155' }}>
                  <span style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'block' }}>Active Collection</span>
                  <code style={{ color: '#4ade80', fontSize: '0.95rem' }}>{crspData.collection}</code>
                </div>
                <div style={{ background: '#1e293b', padding: '1rem', borderRadius: '8px', border: '1px solid #334155' }}>
                  <span style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'block' }}>Total Document Count</span>
                  <span style={{ color: '#818cf8', fontSize: '1.25rem', fontWeight: 'bold' }}>
                    {crspData.total_count?.toLocaleString()}
                  </span>
                </div>
              </div>
              <div style={{ background: '#1e293b', padding: '1rem', borderRadius: '8px', border: '1px solid #334155' }}>
                <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '0.875rem', color: '#cbd5e1' }}>Pipeline Status</h4>
                <p style={{ margin: 0, fontSize: '0.8125rem', color: '#94a3b8', lineHeight: 1.5 }}>
                  Connected successfully to internal cluster MongoDB service. Ready for high-throughput feature aggregation, return distribution computations, and causal analytics pipelines.
                </p>
              </div>
            </div>
          )}

          {activeTab === 'samples' && crspData && (
            <div>
              <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', fontWeight: 600, color: '#cbd5e1' }}>
                CRSP Monthly Stock Sample Records
              </h3>
              <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', overflowX: 'auto', maxHeight: '350px' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem', textAlign: 'left' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #334155', color: '#94a3b8' }}>
                      <th style={{ padding: '0.75rem' }}>#</th>
                      <th style={{ padding: '0.75rem' }}>JSON Payload</th>
                    </tr>
                  </thead>
                  <tbody>
                    {crspData.sample_records?.map((record, index) => (
                      <tr key={index} style={{ borderBottom: '1px solid #1e293b' }}>
                        <td style={{ padding: '0.75rem', color: '#64748b' }}>{index + 1}</td>
                        <td style={{ padding: '0.75rem', fontFamily: 'monospace', color: '#38bdf8', maxWidth: '650px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {JSON.stringify(record)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {activeTab === 'analytics' && (
            <div>
              <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', fontWeight: 600, color: '#cbd5e1' }}>
                Statistical & Scientific Analysis View
              </h3>
              <div style={{ background: '#1e293b', padding: '1.5rem', borderRadius: '8px', border: '1px solid #334155', textAlign: 'center' }}>
                <p style={{ margin: '0 0 0.5rem 0', color: '#e2e8f0', fontWeight: 600 }}>Ready for Advanced Modeling</p>
                <p style={{ margin: 0, fontSize: '0.875rem', color: '#94a3b8' }}>
                  This space is prepared for time-series return distributions, volatility metrics, and causal discovery queries.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
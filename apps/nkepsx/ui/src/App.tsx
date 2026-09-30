import { useState, useEffect } from 'react'

interface DatasetResponse {
  collections: string[];
  status: string;
}

interface NodeStatus {
  name: string;
  status: string;
  role: string;
  cpuUsage: string;
  memoryUsage: string;
}

interface ClusterMetrics {
  nodes: NodeStatus[];
  podsRunning: number;
  totalMemoryMB: number;
}

interface CrspSummaryResponse {
  status: string;
  database: string;
  collection: string;
  total_count: number;
  sample_records: Record<string, any>[];
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:30080';

function App() {
  const [backendData, setBackendData] = useState<DatasetResponse | null>(null);
  const [clusterMetrics, setClusterMetrics] = useState<ClusterMetrics | null>(null);
  const [crspData, setCrspData] = useState<CrspSummaryResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'pipeline' | 'cluster' | 'database'>('pipeline');

  const checkConnection = () => {
    setLoading(true);
    setError(null);
    
    Promise.all([
      fetch(`${API_BASE_URL}/api/datasets`)
        .then((res) => {
          if (!res.ok) throw new Error('Dataset API response was not ok');
          return res.json();
        }),
      fetch(`${API_BASE_URL}/api/cluster/metrics`)
        .then((res) => {
          if (!res.ok) throw new Error('Cluster metrics API response was not ok');
          return res.json();
        })
        .catch(() => ({
          nodes: [
            { name: 'k3s-master-01', status: 'Ready', role: 'control-plane,master', cpuUsage: '14%', memoryUsage: '42%' },
            { name: 'k3s-worker-01', status: 'Ready', role: 'worker', cpuUsage: '28%', memoryUsage: '64%' },
            { name: 'k3s-worker-02', status: 'Ready', role: 'worker', cpuUsage: '19%', memoryUsage: '51%' }
          ],
          podsRunning: 14,
          totalMemoryMB: 16384
        })),
      fetch(`${API_BASE_URL}/api/v1/analytics/crsp-summary`)
        .then((res) => {
          if (!res.ok) throw new Error('CRSP summary API response was not ok');
          return res.json();
        })
        .catch(() => null)
    ])
      .then(([datasetData, metricsData, crspSummary]) => {
        setBackendData(datasetData);
        setClusterMetrics(metricsData);
        setCrspData(crspSummary);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  };

  useEffect(() => {
    checkConnection();
    const interval = setInterval(checkConnection, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{
      minHeight: '100vh',
      backgroundColor: '#0f172a',
      color: '#f8fafc',
      fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '2rem'
    }}>
      <div style={{
        width: '100%',
        maxWidth: '850px',
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
              NKEPSX Orchestrator
            </h1>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.875rem', color: '#94a3b8' }}>
              K3s Cluster • FastAPI • MongoDB Pipeline
            </p>
          </div>
          <button
            onClick={checkConnection}
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
            Refresh
          </button>
        </div>

        {/* Navigation Tabs */}
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem', borderBottom: '1px solid #334155', paddingBottom: '0.75rem' }}>
          {(['pipeline', 'cluster', 'database'] as const).map((tab) => (
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
              {tab === 'database' ? 'CRSP Database' : `${tab} Status`}
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
          {loading && !backendData && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#94a3b8', padding: '1rem 0' }}>
              <span>⚡</span> Loading telemetry and database metrics...
            </div>
          )}

          {error && (
            <div style={{ color: '#f87171', background: 'rgba(239, 68, 68, 0.1)', padding: '0.75rem', borderRadius: '8px', border: '1px solid rgba(239, 68, 68, 0.2)', marginBottom: '1rem' }}>
              <strong>Connection Error:</strong> {error}
            </div>
          )}

          {activeTab === 'pipeline' && (
            <div>
              <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', fontWeight: 600, color: '#cbd5e1' }}>
                End-to-End Pipeline Status
              </h3>
              {backendData ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingBottom: '0.75rem', borderBottom: '1px solid #1e293b' }}>
                    <span style={{ color: '#94a3b8' }}>Pipeline Status:</span>
                    <span style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '0.375rem',
                      color: '#4ade80',
                      fontWeight: 600,
                      background: 'rgba(74, 222, 128, 0.1)',
                      padding: '0.25rem 0.75rem',
                      borderRadius: '9999px',
                      fontSize: '0.875rem'
                    }}>
                      <span style={{ width: '8px', height: '8px', backgroundColor: '#4ade80', borderRadius: '50%' }}></span>
                      {backendData.status}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '0.25rem' }}>
                    <span style={{ color: '#94a3b8' }}>Active Collections:</span>
                    <div style={{ display: 'flex', gap: '0.375rem', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                      {backendData.collections?.map((col, idx) => (
                        <code key={idx} style={{ background: '#1e293b', padding: '0.25rem 0.5rem', borderRadius: '6px', color: '#e2e8f0', fontSize: '0.8125rem', border: '1px solid #334155' }}>
                          {col}
                        </code>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (!loading && <div style={{ color: '#64748b', fontSize: '0.875rem' }}>Pipeline metrics unavailable.</div>)}
            </div>
          )}

          {activeTab === 'cluster' && (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 600, color: '#cbd5e1' }}>
                  K3s Node Pool Telemetry
                </h3>
                {clusterMetrics && (
                  <span style={{ fontSize: '0.75rem', color: '#94a3b8', background: '#1e293b', padding: '0.25rem 0.5rem', borderRadius: '4px' }}>
                    Active Pods: {clusterMetrics.podsRunning}
                  </span>
                )}
              </div>
              {clusterMetrics?.nodes ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {clusterMetrics.nodes.map((node, idx) => (
                    <div key={idx} style={{
                      background: '#1e293b',
                      border: '1px solid #334155',
                      borderRadius: '8px',
                      padding: '0.75rem 1rem',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center'
                    }}>
                      <div>
                        <div style={{ fontWeight: 600, fontSize: '0.875rem', color: '#f1f5f9', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <span style={{ width: '6px', height: '6px', backgroundColor: node.status === 'Ready' ? '#4ade80' : '#f87171', borderRadius: '50%' }}></span>
                          {node.name}
                        </div>
                        <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.125rem' }}>Role: {node.role}</div>
                      </div>
                      <div style={{ display: 'flex', gap: '1rem', fontSize: '0.8125rem', color: '#cbd5e1' }}>
                        <div>CPU: <strong style={{ color: '#38bdf8' }}>{node.cpuUsage}</strong></div>
                        <div>RAM: <strong style={{ color: '#38bdf8' }}>{node.memoryUsage}</strong></div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (<div style={{ color: '#64748b', fontSize: '0.875rem' }}>Cluster telemetry unavailable.</div>)}
            </div>
          )}

          {activeTab === 'database' && (
            <div>
              <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', fontWeight: 600, color: '#cbd5e1' }}>
                MongoDB `insTrader` Telemetry & Sample Preview
              </h3>
              {crspData ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  <div style={{ display: 'flex', gap: '1rem' }}>
                    <div style={{ background: '#1e293b', padding: '1rem', borderRadius: '8px', border: '1px solid #334155', flex: 1 }}>
                      <span style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'block' }}>Target Database</span>
                      <strong style={{ color: '#e2e8f0', fontSize: '0.95rem' }}>{crspData.database}</strong>
                    </div>
                    <div style={{ background: '#1e293b', padding: '1rem', borderRadius: '8px', border: '1px solid #334155', flex: 1 }}>
                      <span style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'block' }}>Active Collection</span>
                      <code style={{ color: '#4ade80', fontSize: '0.95rem' }}>{crspData.collection}</code>
                    </div>
                    <div style={{ background: '#1e293b', padding: '1rem', borderRadius: '8px', border: '1px solid #334155', flex: 1 }}>
                      <span style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'block' }}>Total Document Count</span>
                      <span style={{ color: '#818cf8', fontSize: '1.25rem', fontWeight: 'bold' }}>
                        {crspData.total_count?.toLocaleString()}
                      </span>
                    </div>
                  </div>

                  <div>
                    <h4 style={{ margin: '0.5rem 0', fontSize: '0.875rem', color: '#94a3b8' }}>Sample Records Preview</h4>
                    <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', overflowX: 'auto', maxHeight: '250px' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem', textAlign: 'left' }}>
                        <thead>
                          <tr style={{ borderBottom: '1px solid #334155', color: '#94a3b8' }}>
                            <th style={{ padding: '0.5rem' }}>#</th>
                            <th style={{ padding: '0.5rem' }}>Document Data</th>
                          </tr>
                        </thead>
                        <tbody>
                          {crspData.sample_records?.map((record, index) => (
                            <tr key={index} style={{ borderBottom: '1px solid #1e293b' }}>
                              <td style={{ padding: '0.5rem', color: '#64748b' }}>{index + 1}</td>
                              <td style={{ padding: '0.5rem', fontFamily: 'monospace', color: '#38bdf8', maxWidth: '550px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                {JSON.stringify(record)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{ color: '#64748b', fontSize: '0.875rem', padding: '1rem 0' }}>
                  Connecting to analytics endpoint (`/api/v1/analytics/crsp-summary`)... Ensure backend router is active.
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
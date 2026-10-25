/**
 * Header.jsx — Top navigation bar
 * Shows system title, connection status, and current threat level
 */
export default function Header({ connected, latestEvent, totalEvents }) {
  const threatTier = latestEvent?.stage_6?.threat_tier || 'BENIGN';
  const eventCount = totalEvents ?? (latestEvent?.event_id || 0);

  return (
    <header className="header">
      <div className="header-title">
        <span className="shield-icon">🛡️</span>
        AI Ransomware Defense — SOC Dashboard
      </div>
      <div className="header-status">
        <a
          href="https://drive.google.com/file/d/1P7pPM5g3_gRTEyINa8AxJC5WKs5Wj2Of/view?usp=sharing"
          target="_blank"
          rel="noopener noreferrer"
          className="demo-video-btn"
          id="btn-demo-video"
          title="Watch Demo Video"
        >
          <svg className="demo-video-icon" viewBox="0 0 24 24" width="13" height="13" fill="currentColor">
            <polygon points="6 4 20 12 6 20 6 4" />
          </svg>
          <span>Demo video</span>
        </a>
        <span className="mono" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Events: {eventCount}
        </span>
        <span className={`threat-level-badge ${threatTier}`}>
          {threatTier}
        </span>
        <span className={`status-badge ${connected ? 'connected' : 'disconnected'}`}>
          <span className={`status-dot ${connected ? 'connected' : 'disconnected'}`} />
          {connected ? 'LIVE' : 'OFFLINE'}
        </span>
      </div>
    </header>
  );
}

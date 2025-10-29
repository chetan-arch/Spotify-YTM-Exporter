import { useEffect, useState } from "react";

// Build API base from current host (works with 127.0.0.1 or localhost)
const API = `http://127.0.0.1:8000`;

export default function App() {
  const [spStatus, setSpStatus] = useState({ connected: false });
  const [ytStatus, setYtStatus] = useState({ connected: false });
  const [exporting, setExporting] = useState(false);
  const [exportResult, setExportResult] = useState(null);
  const [recs, setRecs] = useState([]);
  const [loadingRecs, setLoadingRecs] = useState(false);
  const [playlistName, setPlaylistName] = useState("");

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("spotify") === "connected") {
      window.history.replaceState({}, "", window.location.pathname);
    }
    refreshStatus();
  }, []);

  async function refreshStatus() {
    const [sp, yt] = await Promise.all([
      fetch(`${API}/api/spotify/status`).then(r => r.json()).catch(() => ({connected:false})),
      fetch(`${API}/api/ytmusic/status`).then(r => r.json()).catch(() => ({connected:false})),
    ]);
    setSpStatus(sp);
    setYtStatus(yt);
  }

  async function connectSpotify() {
    try {
      const res = await fetch(`${API}/api/spotify/login`);
      const raw = await res.text();
      let data = null;
      try { data = JSON.parse(raw); } catch {}
      if (!res.ok || !data?.auth_url) {
        throw new Error(data?.detail || raw || `HTTP ${res.status}`);
      }
      window.location.href = data.auth_url;
    } catch (e) {
      alert(`Spotify login failed: ${e.message}`);
    }
  }

  async function connectYTM() {
    alert(
`How to get YouTube Music headers:
1) Open music.youtube.com while logged in (Chrome recommended).
2) Press F12 → Network tab → reload page.
3) Click an XHR like https://music.youtube.com/youtubei/v1/browse
4) Copy ALL “Request headers” and paste as JSON key/values, e.g.:
{
  "cookie": "VISITOR_INFO1_LIVE=...; YSC=...; ...",
  "user-agent": "Mozilla/5.0 ...",
  "x-goog-authuser": "0",
  "x-origin": "https://music.youtube.com"
}
Tip: Ensure the JSON includes a "cookie" key (full line, one line).`
    );
    const raw = prompt("Paste your headers JSON here:");
    if (!raw) return;
    let headersObj = null;
    try {
      headersObj = JSON.parse(raw.trim());
    } catch (e) {
      alert("Invalid JSON. Please try again.");
      return;
    }
    try {
      const res = await fetch(`${API}/api/ytmusic/connect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ headers_json: headersObj }),
      });
      const text = await res.text();
      let data = null;
      try { data = JSON.parse(text); } catch {}
      if (!res.ok) throw new Error((data && (data.detail || data.message)) || text || `HTTP ${res.status}`);
      alert("YouTube Music connected!");
      await refreshStatus();
    } catch (e) {
      alert("Failed to connect YT Music: " + e.message);
    }
  }

  async function doExport() {
    setExporting(true);
    setExportResult(null);
    try {
      const res = await fetch(`${API}/api/export`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          playlist_name: playlistName || null,
          also_like_on_ytm: false,
        }),
      });

      const raw = await res.text();
      let data = null;
      try { data = JSON.parse(raw); } catch {}

      if (!res.ok) {
        const msg = (data && (data.detail || data.message)) || raw || `HTTP ${res.status}`;
        throw new Error(msg);
      }

      setExportResult(data);
    } catch (e) {
      alert(`Export failed: ${e.message}`);
      console.error(e);
    } finally {
      setExporting(false);
    }
  }

  async function loadRecsOffline() {
    setLoadingRecs(true);
    setRecs([]);
    try {
      const res = await fetch(`${API}/api/recommendations_offline`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ count: 30 }),
      });
      const raw = await res.text();
      let data = null;
      try { data = JSON.parse(raw); } catch {}
      if (!res.ok) {
        const msg = (data && (data.detail?.message || data.detail || data.message)) || raw || `HTTP ${res.status}`;
        throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
      }
      setRecs(data.suggestions || []);
    } catch (e) {
      alert("Failed to load recommendations: " + e.message);
      console.error(e);
    } finally {
      setLoadingRecs(false);
    }
  }

  const canExport = spStatus.connected && ytStatus.connected;

  return (
    <div style={{ fontFamily: "Inter, system-ui, sans-serif", padding: 24, maxWidth: 900, margin: "0 auto" }}>
      <h1>Spotify ➜ YouTube Music Exporter</h1>
      <p style={{ opacity: 0.8 }}>
        Connect both accounts, export your <strong>Liked Songs</strong> into a new YouTube Music playlist,
        then get suggestions powered by an offline dataset (no Spotify audio features needed).
      </p>

      <section style={card}>
        <h2>1) Connect Accounts</h2>
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
          <button onClick={connectSpotify} style={btn}>
            {spStatus.connected ? "✅ Spotify Connected" : "Connect Spotify"}
          </button>
          <button onClick={connectYTM} style={btn}>
            {ytStatus.connected ? "✅ YouTube Music Connected" : "Connect YouTube Music"}
          </button>
          <button onClick={refreshStatus} style={btnGhost}>Refresh</button>
        </div>
        <div style={{ marginTop: 8, fontSize: 14 }}>
          {spStatus.connected && <div>Spotify user: <b>{spStatus.user}</b></div>}
          {!ytStatus.connected && ytStatus.reason && (
            <div style={{ color: "#b00", marginTop: 6 }}>YT status: {String(ytStatus.reason)}</div>
          )}
        </div>
      </section>

      <section style={card}>
        <h2>2) Export Liked Songs → YT Music</h2>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <input
            placeholder='Playlist name (optional)'
            value={playlistName}
            onChange={e => setPlaylistName(e.target.value)}
            style={input}
          />
          <button disabled={!canExport || exporting} onClick={doExport} style={btnPrimary}>
            {exporting ? "Exporting..." : "Export to YouTube Music"}
          </button>
        </div>
        {!canExport && <p style={{ color: "#b00", marginTop: 12 }}>Connect both Spotify and YouTube Music to continue.</p>}
        {exportResult && (
          <div style={{ marginTop: 16, lineHeight: 1.7 }}>
            <div><b>Playlist ID:</b> {exportResult.created_playlist_id} &nbsp;
              <a href={`https://music.youtube.com/playlist?list=${exportResult.created_playlist_id}`} target="_blank" rel="noreferrer">
                (open)
              </a>
            </div>
            <div><b>Total liked on Spotify:</b> {exportResult.total_spotify_tracks}</div>
            <div><b>Matched:</b> {exportResult.matched}</div>
            <div><b>Added to playlist:</b> {exportResult.added}</div>
            <div><b>Also liked on YTM:</b> {exportResult.liked}</div>

            {exportResult.unmatched_samples?.length > 0 && (
              <>
                <div><b>Unmatched samples (showing {exportResult.unmatched_samples.length}):</b></div>
                <ul style={{marginTop: 8}}>
                  {exportResult.unmatched_samples.map((t, i) => {
                    const title = t.name || "(no title)";
                    const artists = (t.artists && t.artists.length) ? t.artists.join(", ") : "(no artists)";
                    const diag = t.diagnostic || {};
                    const best = [diag.best_title, (diag.best_artists || []).join(", ")].filter(Boolean).join(" — ");
                    return (
                      <li key={i} style={{marginBottom: 8}}>
                        <div>{title} — {artists}</div>
                        {best && (
                          <div style={{opacity: 0.8, fontSize: 13}}>
                            Closest on YTM: {best} (score {diag.best_score ?? 0})
                          </div>
                        )}
                        {diag.last_query && (
                          <div style={{opacity: 0.7, fontSize: 12}}>
                            Query tried: <code>{diag.last_query}</code>
                          </div>
                        )}
                        {diag.last_error && (
                          <div style={{opacity: 0.7, fontSize: 12, color:"#b55"}}>
                            Note: {diag.last_error}
                          </div>
                        )}
                      </li>
                    );
                  })}
                </ul>
              </>
            )}
          </div>
        )}
      </section>

      <section style={card}>
        <h2>3) Suggestions (offline dataset)</h2>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          <button disabled={!spStatus.connected || loadingRecs} onClick={loadRecsOffline} style={btn}>
            {loadingRecs ? "Crunching..." : "Get Suggestions (offline dataset)"}
          </button>
        </div>
        {!spStatus.connected && <p style={{ color: "#b00", marginTop: 12 }}>Connect Spotify to fetch suggestions.</p>}
        {recs?.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th style={th}>Title</th>
                  <th style={th}>Artists</th>
                  <th style={th}>Album</th>
                  <th style={th}>Score</th>
                </tr>
              </thead>
              <tbody>
                {recs.map((r, idx) => (
                  <tr key={idx}>
                    <td style={td}>{r.name}</td>
                    <td style={td}>{(r.artists || []).join(", ")}</td>
                    <td style={td}>{r.album}</td>
                    <td style={td}>{r.score ?? ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p style={{ marginTop: 8, fontSize: 14, opacity: 0.8 }}>
              These are suggestions only. We can add a “Send selected to YTM” flow next.
            </p>
          </div>
        )}
      </section>
    </div>
  );
}

const card = {
  border: "1px solid #282828",
  borderRadius: 16,
  padding: 16,
  marginTop: 16,
  background: "rgba(255,255,255,0.03)"
};

const btn = {
  padding: "10px 14px",
  borderRadius: 10,
  border: "1px solid #444",
  background: "transparent",
  cursor: "pointer",
  color: "inherit"
};

const btnGhost = { ...btn, opacity: 0.8 };

const btnPrimary = {
  ...btn,
  border: "1px solid #0a84ff",
  background: "linear-gradient(0deg, #0a84ff22, #0a84ff22)",
};

const input = {
  padding: "10px 12px",
  borderRadius: 10,
  border: "1px solid #444",
  outline: "none",
  background: "transparent",
  color: "inherit",
  width: 300
};

const th = { textAlign: "left", borderBottom: "1px solid #333", padding: 8 };
const td = { borderBottom: "1px solid #222", padding: 8 };

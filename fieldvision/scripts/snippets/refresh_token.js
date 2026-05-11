// Paste this into Chrome DevTools Console on any logged-in mlb.com page.
// Saves the current api://mlb_default access token to ~/Downloads/fv_token.txt.
// Then run:  mv ~/Downloads/fv_token.txt ~/fieldvision/.fv_token.txt
// to install it. The daemon picks it up automatically on its next poll.
(() => {
  const raw = localStorage.getItem('okta-token-storage');
  const RX = /eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/g;
  let token = null;
  for (const j of new Set(raw.match(RX) || [])) {
    try {
      const c = JSON.parse(atob(j.split('.')[1].replace(/-/g,'+').replace(/_/g,'/')));
      if (c.aud === 'api://mlb_default') { token = j; break; }
    } catch(e) {}
  }
  if (!token) { console.error('No api://mlb_default token found.'); return; }
  const blob = new Blob([token], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'fv_token.txt';
  document.body.appendChild(a); a.click(); a.remove();
  console.log('%c✓ Token saved to ~/Downloads/fv_token.txt', 'color:lime;font-weight:bold');
  console.log('Now run: mv ~/Downloads/fv_token.txt ~/fieldvision/.fv_token.txt');
})();

import { ClerkProvider, SignIn, UserButton, useAuth } from "@clerk/react";
import { createRoot } from "react-dom/client";
import { api, setTokenGetter } from "./api";
import App from "./App";
import { Buddy } from "./Room";
import "./styles.css";

/** Signed out: the login screen. Signed in: the app, with every API call carrying the session token. */
function Gate() {
  const { isLoaded, isSignedIn, getToken } = useAuth();
  if (!isLoaded) return <div className="splash">Opening the clubhouse…</div>;
  if (!isSignedIn) {
    return (
      <div className="login">
        <div className="login-hero">
          <div className="login-buddies">
            <Buddy hue={265} state="thinking" /><Buddy hue={42} state="talking" boss /><Buddy hue={165} state="idle" />
          </div>
          <h1>Atrium</h1>
          <p>Build your own team of AI agents and watch them work.</p>
        </div>
        <SignIn routing="hash" />
      </div>
    );
  }
  setTokenGetter(() => getToken());
  return <App account={<UserButton />} />;
}

async function boot() {
  const root = createRoot(document.getElementById("root")!);
  root.render(<div className="splash">Opening the clubhouse…</div>);
  try {
    const cfg = await api.publicConfig();
    if (cfg.clerk_publishable_key) {
      root.render(<ClerkProvider publishableKey={cfg.clerk_publishable_key}><Gate /></ClerkProvider>);
    } else {
      root.render(<App account={<span className="pill warn">dev mode · no login</span>} />);
    }
  } catch {
    root.render(<div className="splash">Cannot reach the Atrium server. Is the backend running?</div>);
  }
}

boot();

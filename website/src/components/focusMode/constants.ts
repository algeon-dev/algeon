// Shared by the browser components and `docusaurus.config.ts` (Node), so this
// module holds plain values only.

/** Attribute on `<html>` that drives every focus mode rule in CSS. */
export const FOCUS_MODE_ATTRIBUTE = 'data-focus-mode';

/** Session-scoped: a reload restores focus mode, a new tab starts without it. */
export const FOCUS_MODE_STORAGE_KEY = 'algeon-focus-mode';

/** Runs in `<head>` before the first paint so a reload never shows the chrome. */
export const FOCUS_MODE_RESTORE_SCRIPT = `try{if(sessionStorage.getItem('${FOCUS_MODE_STORAGE_KEY}')==='on')document.documentElement.setAttribute('${FOCUS_MODE_ATTRIBUTE}','on')}catch(e){}`;

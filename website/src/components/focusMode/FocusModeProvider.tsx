import React, {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

import {FOCUS_MODE_ATTRIBUTE, FOCUS_MODE_STORAGE_KEY} from './constants';

// Set by DocItem/Layout on the container that is centered, focused and
// animated on a switch.
const CONTENT_SELECTOR = '[data-focus-mode-content]';
const NAVBAR_SELECTOR = '.theme-layout-navbar';
const SEARCH_INPUT_SELECTOR = '.navbar__search-input';
const READING_BLOCK_SELECTOR =
  'h1, h2, h3, h4, h5, h6, p, li, pre, table, blockquote, figure';
const VIEW_TRANSITION_NAME = 'focus-mode-content';
const IDLE_ATTRIBUTE = 'data-focus-mode-idle';
// Focus mode controls stay at full opacity for this long after entering and
// after each mouse move, then CSS dims them.
const AWAKE_MS = 3000;

type FocusModeContextValue = {
  active: boolean;
  setActive: (next: boolean) => void;
  /** Text for the polite live region; empty until the first switch. */
  announcement: string;
};

const FocusModeContext = createContext<FocusModeContextValue | null>(null);

export function useFocusMode(): FocusModeContextValue {
  const context = useContext(FocusModeContext);
  if (!context) {
    throw new Error('useFocusMode must be used inside FocusModeProvider');
  }
  return context;
}

// Blog posts have no marked container; their `main` holds only the post.
function contentElement(): HTMLElement | null {
  return (
    document.querySelector<HTMLElement>(CONTENT_SELECTOR) ??
    document.querySelector<HTMLElement>('main')
  );
}

// The navbar is sticky, so anything above its bottom edge is covered. A hidden
// navbar measures 0.
function navbarBottom(): number {
  const navbar = document.querySelector(NAVBAR_SELECTOR);
  return navbar ? Math.max(0, navbar.getBoundingClientRect().bottom) : 0;
}

function readingAnchor(content: HTMLElement): HTMLElement | null {
  const visibleTop = navbarBottom();
  for (const block of content.querySelectorAll<HTMLElement>(
    READING_BLOCK_SELECTOR,
  )) {
    if (block.getBoundingClientRect().bottom > visibleTop) {
      return block;
    }
  }
  return null;
}

// tabindex -1 lets the container take focus without joining the tab order.
// It stays: Chrome blurs an element whose tabindex is removed while focused.
function focusContent(content: HTMLElement): void {
  content.setAttribute('tabindex', '-1');
  content.focus({preventScroll: true});
}

function isEditable(target: EventTarget | null): boolean {
  return (
    target instanceof HTMLElement &&
    (target.isContentEditable || target.matches('input, textarea'))
  );
}

function hasMod(event: KeyboardEvent): boolean {
  return (event.metaKey || event.ctrlKey) && !event.altKey;
}

export function FocusModeProvider({children}: {children: ReactNode}): ReactNode {
  const [active, setActiveState] = useState(false);
  const [announcement, setAnnouncement] = useState('');
  // The latest requested state. It leads `active` while a view transition
  // waits for its update callback.
  const requested = useRef(false);

  useEffect(() => {
    const restored =
      document.documentElement.getAttribute(FOCUS_MODE_ATTRIBUTE) === 'on';
    requested.current = restored;
    setActiveState(restored);
  }, []);

  const apply = useCallback((next: boolean) => {
    const root = document.documentElement;
    const content = contentElement();
    // At the top of the page the page stays at the top.
    const anchor = window.scrollY > 0 && content ? readingAnchor(content) : null;
    const anchorTop = anchor?.getBoundingClientRect().top ?? 0;

    if (next) {
      root.setAttribute(FOCUS_MODE_ATTRIBUTE, 'on');
    } else {
      root.removeAttribute(FOCUS_MODE_ATTRIBUTE);
    }
    try {
      if (next) {
        sessionStorage.setItem(FOCUS_MODE_STORAGE_KEY, 'on');
      } else {
        sessionStorage.removeItem(FOCUS_MODE_STORAGE_KEY);
      }
    } catch {
      // Storage can be unavailable; the switch still applies to this page view.
    }
    setActiveState(next);
    setAnnouncement(
      next ? 'Focus mode on. Press Escape to exit.' : 'Focus mode off.',
    );

    // Safari has no scroll anchoring, so keep the reading position by hand.
    // Reading the rect forces layout of the new state. On exit, a block that
    // starts in the band the navbar returns to moves just below it.
    if (anchor) {
      const target =
        !next && anchorTop >= 0 ? Math.max(anchorTop, navbarBottom()) : anchorTop;
      window.scrollBy({
        top: anchor.getBoundingClientRect().top - target,
        behavior: 'instant',
      });
    }

    const focused = document.activeElement;
    if (
      content &&
      (next || (focused instanceof HTMLElement && focused.getClientRects().length === 0))
    ) {
      focusContent(content);
    }
  }, []);

  const switchTo = useCallback(
    (next: boolean, animate: boolean) => {
      if (requested.current === next) {
        return;
      }
      requested.current = next;
      if (
        !animate ||
        !('startViewTransition' in document) ||
        window.matchMedia('(prefers-reduced-motion: reduce)').matches
      ) {
        apply(next);
        return;
      }
      const content = contentElement();
      content?.style.setProperty('view-transition-name', VIEW_TRANSITION_NAME);
      const transition = document.startViewTransition(() => apply(next));
      // A newer switch skips this animation; its update has still run.
      transition.ready.catch(() => {});
      transition.finished.finally(() =>
        content?.style.removeProperty('view-transition-name'),
      );
    },
    [apply],
  );

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.isComposing || event.repeat) {
        return;
      }
      if (hasMod(event) && event.shiftKey && event.code === 'KeyF') {
        if (isEditable(event.target)) {
          return;
        }
        event.preventDefault();
        switchTo(!requested.current, true);
      } else if (
        event.key === 'Escape' &&
        requested.current &&
        !event.defaultPrevented
      ) {
        switchTo(false, true);
      }
    };
    // Capture phase on document runs before the search plugin's own mod+k
    // listener there, which would otherwise focus an input inside the hidden
    // navbar and fail without feedback.
    const onSearchShortcut = (event: KeyboardEvent) => {
      if (
        requested.current &&
        hasMod(event) &&
        !event.shiftKey &&
        event.code === 'KeyK'
      ) {
        event.preventDefault();
        switchTo(false, false);
        document.querySelector<HTMLInputElement>(SEARCH_INPUT_SELECTOR)?.focus();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    document.addEventListener('keydown', onSearchShortcut, true);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      document.removeEventListener('keydown', onSearchShortcut, true);
    };
  }, [switchTo]);

  useEffect(() => {
    if (!active) {
      return;
    }
    const root = document.documentElement;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const wake = () => {
      root.removeAttribute(IDLE_ATTRIBUTE);
      clearTimeout(timer);
      timer = setTimeout(() => root.setAttribute(IDLE_ATTRIBUTE, ''), AWAKE_MS);
    };
    wake();
    window.addEventListener('mousemove', wake, {passive: true});
    return () => {
      clearTimeout(timer);
      window.removeEventListener('mousemove', wake);
      root.removeAttribute(IDLE_ATTRIBUTE);
    };
  }, [active]);

  const value = useMemo(
    () => ({
      active,
      setActive: (next: boolean) => switchTo(next, true),
      announcement,
    }),
    [active, announcement, switchTo],
  );

  return (
    <FocusModeContext.Provider value={value}>{children}</FocusModeContext.Provider>
  );
}

import React, {type ReactNode} from 'react';

import {FocusModeProvider} from '@site/src/components/focusMode/FocusModeProvider';
import FocusModeExit from '@site/src/components/focusMode/FocusModeExit';

// Root stays mounted across client-side navigation, so focus mode survives it.
// The exit button precedes the page in DOM order: with the chrome hidden, one
// Shift+Tab from the content reaches it.
export default function Root({children}: {children: ReactNode}): ReactNode {
  return (
    <FocusModeProvider>
      <FocusModeExit />
      {children}
    </FocusModeProvider>
  );
}

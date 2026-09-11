/**
 * Mobile navbar drawer.
 *
 * Replaces Docusaurus's two-level drawer (primary menu, then a sliding docs
 * panel behind "Back to main menu") with one scrollable panel: the navbar
 * links and a color-mode row first, then the current docs sidebar. The header
 * keeps only the logo and close button; the color-mode switch lives in the
 * menu so it never competes with the wide wordmark for header width.
 */
import React, {type ReactNode} from 'react';
import clsx from 'clsx';
import {
  ThemeClassNames,
  useColorMode,
  useThemeConfig,
} from '@docusaurus/theme-common';
import {
  useLockBodyScroll,
  useNavbarMobileSidebar,
  useNavbarSecondaryMenu,
} from '@docusaurus/theme-common/internal';
import {translate} from '@docusaurus/Translate';
import IconClose from '@theme/Icon/Close';
import IconDarkMode from '@theme/Icon/DarkMode';
import IconLightMode from '@theme/Icon/LightMode';
import IconSystemColorMode from '@theme/Icon/SystemColorMode';
import NavbarItem, {type Props as NavbarItemConfig} from '@theme/NavbarItem';
import NavbarLogo from '@theme/Navbar/Logo';

type ColorModeChoice = 'light' | 'dark' | null;

function nextColorMode(
  choice: ColorModeChoice,
  respectPrefersColorScheme: boolean,
): ColorModeChoice {
  if (!respectPrefersColorScheme) {
    return choice === 'dark' ? 'light' : 'dark';
  }
  switch (choice) {
    case null:
      return 'light';
    case 'light':
      return 'dark';
    default:
      return null;
  }
}

function colorModeLabel(choice: ColorModeChoice): string {
  switch (choice) {
    case 'light':
      return translate({
        id: 'theme.colorToggle.ariaLabel.mode.light',
        message: 'light mode',
        description: 'The name for the light color mode',
      });
    case 'dark':
      return translate({
        id: 'theme.colorToggle.ariaLabel.mode.dark',
        message: 'dark mode',
        description: 'The name for the dark color mode',
      });
    default:
      return translate({
        id: 'theme.colorToggle.ariaLabel.mode.system',
        message: 'system mode',
        description: 'The name for the system color mode',
      });
  }
}

function ColorModeMenuItem(): ReactNode {
  const {disableSwitch, respectPrefersColorScheme} = useThemeConfig().colorMode;
  const {colorModeChoice, setColorMode} = useColorMode();
  if (disableSwitch) {
    return null;
  }
  const label = colorModeLabel(colorModeChoice);
  return (
    <li className="menu__list-item">
      <button
        type="button"
        className="clean-btn menu__link nx-mobile-drawer__color-mode"
        aria-label={translate(
          {
            id: 'theme.colorToggle.ariaLabel',
            message: 'Switch between dark and light mode (currently {mode})',
            description: 'The ARIA label for the color mode toggle',
          },
          {mode: label},
        )}
        onClick={() =>
          setColorMode(nextColorMode(colorModeChoice, respectPrefersColorScheme))
        }>
        {/* All three icons render so the right one shows before hydration;
            CSS picks by the root's data-theme-choice attribute. */}
        <IconLightMode aria-hidden className="nx-mobile-drawer__color-mode-icon nx-mobile-drawer__color-mode-icon--light" />
        <IconDarkMode aria-hidden className="nx-mobile-drawer__color-mode-icon nx-mobile-drawer__color-mode-icon--dark" />
        <IconSystemColorMode aria-hidden className="nx-mobile-drawer__color-mode-icon nx-mobile-drawer__color-mode-icon--system" />
        <span>
          {translate(
            {
              id: 'algeon.mobileDrawer.colorMode',
              message: 'Theme: {mode}',
              description: 'Color mode row label in the mobile navbar drawer',
            },
            {mode: label},
          )}
        </span>
      </button>
    </li>
  );
}

function CloseButton(): ReactNode {
  const mobileSidebar = useNavbarMobileSidebar();
  return (
    <button
      type="button"
      aria-label={translate({
        id: 'theme.docs.sidebar.closeSidebarButtonAriaLabel',
        message: 'Close navigation bar',
        description: 'The ARIA label for close button of mobile sidebar',
      })}
      className="clean-btn navbar-sidebar__close"
      onClick={() => mobileSidebar.toggle()}>
      <IconClose color="var(--ifm-color-emphasis-600)" />
    </button>
  );
}

function PrimaryLinks(): ReactNode {
  const mobileSidebar = useNavbarMobileSidebar();
  // Same cast as theme-classic: ThemeConfig types navbar items loosely.
  const items = useThemeConfig().navbar.items as NavbarItemConfig[];
  return (
    <ul className="menu__list">
      {items.map((item, i) => (
        <NavbarItem
          mobile
          {...item}
          onClick={() => mobileSidebar.toggle()}
          key={i}
        />
      ))}
      <ColorModeMenuItem />
    </ul>
  );
}

export default function NavbarMobileSidebar(): ReactNode {
  const mobileSidebar = useNavbarMobileSidebar();
  const docsMenu = useNavbarSecondaryMenu().content;
  useLockBodyScroll(mobileSidebar.shown);
  if (!mobileSidebar.shouldRender) {
    return null;
  }
  return (
    <div
      className={clsx(
        ThemeClassNames.layout.navbar.mobileSidebar.container,
        'navbar-sidebar',
      )}>
      <div className="navbar-sidebar__brand">
        <NavbarLogo />
        <CloseButton />
      </div>
      <div className="navbar-sidebar__items">
        <div
          className={clsx(
            ThemeClassNames.layout.navbar.mobileSidebar.panel,
            'navbar-sidebar__item menu',
          )}>
          <PrimaryLinks />
          {docsMenu && (
            <>
              <hr className="nx-mobile-drawer__separator" />
              {docsMenu}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

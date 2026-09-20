import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

// Theme `<TabItem>` panels stay mounted with the `hidden` attribute, so a
// same-page link to a heading inside an unselected tab (for example a TOC entry)
// has no box to scroll to. Selecting the owning tab in the capture phase lets
// React commit before the browser runs the link's default fragment navigation.
function revealAnchorTarget(event: MouseEvent): void {
  if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
    return;
  }
  const anchor = (event.target as Element | null)?.closest?.('a[href*="#"]');
  if (!(anchor instanceof HTMLAnchorElement) || anchor.pathname !== window.location.pathname) {
    return;
  }
  const target = document.getElementById(decodeURIComponent(anchor.hash.slice(1)));
  let panel = target?.closest('[role="tabpanel"][hidden]');
  while (panel) {
    // Tabs renders its tab list and panel container as siblings, with one
    // panel per tab in the same order.
    const panels = Array.from(panel.parentElement!.children).filter(
      (element) => element.getAttribute('role') === 'tabpanel',
    );
    const tabs = panel.parentElement!.parentElement!.querySelectorAll(':scope > [role="tablist"] > [role="tab"]');
    (tabs[panels.indexOf(panel)] as HTMLElement | undefined)?.click();
    panel = panel.parentElement!.closest('[role="tabpanel"][hidden]');
  }
}

if (ExecutionEnvironment.canUseDOM) {
  document.addEventListener('click', revealAnchorTarget, true);
}

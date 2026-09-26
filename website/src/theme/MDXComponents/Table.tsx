import React, {type ComponentProps, type ReactNode} from 'react';
import clsx from 'clsx';

function getColumnCount(children: ReactNode): number {
  let count = 0;
  React.Children.forEach(children, (child) => {
    if (!React.isValidElement(child)) {
      return;
    }
    const {type, props} = child as {type: any; props?: {children?: ReactNode}};
    if (type === 'thead' || type === 'tbody') {
      React.Children.forEach(props?.children, (row) => {
        if (count === 0 && React.isValidElement(row)) {
          const cells = (row as {props?: {children?: ReactNode}}).props?.children;
          count = React.Children.count(cells);
        }
      });
    }
  });
  return count;
}

export default function MDXTable(props: ComponentProps<'table'>) {
  const colCount = getColumnCount(props.children);
  const isCompact = colCount > 0 && colCount <= 2;
  const isWide = colCount >= 5;

  return (
    <div
      className={clsx(
        'table-responsive-wrapper',
        'thin-scrollbar',
        isCompact && 'table-responsive-wrapper--compact',
        isWide && 'table-responsive-wrapper--wide',
      )}
      data-col-count={colCount || undefined}
    >
      <table {...props} />
    </div>
  );
}

import {createContext} from 'react';
import type {FoldRange} from '@site/src/utils/sqlFolds';

export const FoldContext = createContext<FoldRange[]>([]);

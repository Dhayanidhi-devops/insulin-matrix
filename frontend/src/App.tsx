import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { AnnAnalyticsPage } from './pages/AnnAnalytics'
import { AssessmentPage } from './pages/Assessment'
import { BatchPage } from './pages/Batch'
import { ComparisonPage } from './pages/Comparison'
import { DatasetPage } from './pages/Dataset'
import { HistoryPage } from './pages/History'
import { OverviewPage } from './pages/Overview'
import { PreprocessingPage } from './pages/Preprocessing'
import { ReportsPage } from './pages/Reports'
import { SettingsPage } from './pages/Settings'
import { TrainingPage } from './pages/Training'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<OverviewPage />} />
          <Route path="assessment" element={<AssessmentPage />} />
          <Route path="dataset" element={<DatasetPage />} />
          <Route path="preprocessing" element={<PreprocessingPage />} />
          <Route path="training" element={<TrainingPage />} />
          <Route path="comparison" element={<ComparisonPage />} />
          <Route path="ann" element={<AnnAnalyticsPage />} />
          <Route path="batch" element={<BatchPage />} />
          <Route path="reports" element={<ReportsPage />} />
          <Route path="history" element={<HistoryPage />} />
          <Route path="settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

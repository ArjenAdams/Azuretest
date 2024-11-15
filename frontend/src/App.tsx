import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SADBarTest from './pages/test/SADBarTest';

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/test/button" element={<ButtonTestPage />} />
        <Route path="/test/sadbar" element={<SADBarTest />} />
      </Routes>
    </Router>
  );
}

export default App;

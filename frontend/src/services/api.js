

const API_BASE_URL = 'http://localhost:5000'; // Your Flask backend URL

export async function startResearch(query) {
  const response = await fetch(`${API_BASE_URL}/research`, {
    method: 'POST',
    credentials: 'include' ,// Important for CORS with credentials if needed
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query }),
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ message: "Unknown error occurred" }));
    throw new Error(errorData.error || errorData.message || `HTTP error! status: ${response.status}`);
  }
  return response.json();
}

export async function getResearchStatus(projectId) {
  const response = await fetch(`${API_BASE_URL}/research_status/${projectId}`);
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ message: "Unknown error occurred" }));
    throw new Error(errorData.error || errorData.message || `HTTP error! status: ${response.status}`);
  }
  return response.json();
}
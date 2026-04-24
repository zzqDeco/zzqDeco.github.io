type Repo = {
  name: string;
  description: string;
  url: string;
  language?: string;
  languageColor?: string;
  stars: number;
  forks: number;
  updatedAt?: string;
};

declare const process: {
  env: {
    GITHUB_TOKEN?: string;
  };
};

export type GitHubProfile = {
  login: string;
  name: string;
  bio: string;
  avatarUrl: string;
  url: string;
  company: string;
  location: string;
  followers: number;
  following: number;
  publicRepos: number;
  stars: number;
  repos: Repo[];
};

const fallbackProfile: GitHubProfile = {
  login: 'zzqDeco',
  name: 'zzqDeco',
  bio: 'Full-Stack AI Engineer & Researcher\nBuilding only what I truly wanna build\nHangzhou Dianzi University',
  avatarUrl: 'https://avatars.githubusercontent.com/u/41999232?v=4',
  url: 'https://github.com/zzqDeco',
  company: 'Hangzhou Dianzi University',
  location: 'Hangzhou China',
  followers: 2,
  following: 6,
  publicRepos: 7,
  stars: 28,
  repos: [
    {
      name: 'starxo',
      description: 'AI Sandbox Coding Agent Desktop App',
      url: 'https://github.com/zzqDeco/starxo',
      language: 'Go',
      stars: 0,
      forks: 0,
    },
    {
      name: 'shadiff',
      description: 'Shadow traffic semantic comparison for API migration validation',
      url: 'https://github.com/zzqDeco/shadiff',
      language: 'Go',
      stars: 0,
      forks: 0,
    },
    {
      name: 'papersilm',
      description: 'Paper-focused Document Agent CLI',
      url: 'https://github.com/zzqDeco/papersilm',
      language: 'Go',
      stars: 0,
      forks: 0,
    },
  ],
};

const userAgent = 'zzqDeco.github.io';
const pinnedRepoLimit = 6;

const toRepo = (repo: any): Repo => ({
  name: repo.name,
  description: repo.description ?? 'No description provided.',
  url: repo.url ?? repo.html_url,
  language: repo.primaryLanguage?.name ?? repo.language ?? undefined,
  languageColor: repo.primaryLanguage?.color ?? undefined,
  stars: repo.stargazerCount ?? repo.stargazers_count ?? 0,
  forks: repo.forkCount ?? repo.forks_count ?? 0,
  updatedAt: repo.updatedAt ?? repo.updated_at,
});

const parseLastPage = (linkHeader: string | null): number | undefined => {
  if (!linkHeader) return undefined;
  const match = linkHeader.match(/[?&]page=(\d+)>;\s*rel="last"/);
  return match ? Number(match[1]) : undefined;
};

const fetchGraphQLProfile = async (token: string): Promise<GitHubProfile> => {
  const response = await fetch('https://api.github.com/graphql', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
      'User-Agent': userAgent,
    },
    body: JSON.stringify({
      query: `
        query Profile($login: String!) {
          user(login: $login) {
            login
            name
            bio
            avatarUrl
            url
            company
            location
            followers {
              totalCount
            }
            following {
              totalCount
            }
            repositories(first: 100, ownerAffiliations: OWNER, privacy: PUBLIC) {
              totalCount
            }
            starredRepositories {
              totalCount
            }
            pinnedItems(first: ${pinnedRepoLimit}, types: REPOSITORY) {
              nodes {
                ... on Repository {
                  name
                  description
                  url
                  stargazerCount
                  forkCount
                  updatedAt
                  primaryLanguage {
                    name
                    color
                  }
                }
              }
            }
          }
        }
      `,
      variables: { login: 'zzqDeco' },
    }),
  });

  if (!response.ok) {
    throw new Error(`GraphQL profile request failed: ${response.status}`);
  }

  const payload = await response.json();
  const user = payload?.data?.user;
  if (!user) {
    throw new Error('GraphQL profile response did not include a user.');
  }

  const pinnedRepos = (user.pinnedItems?.nodes ?? []).map(toRepo);

  return {
    login: user.login ?? fallbackProfile.login,
    name: user.name ?? user.login ?? fallbackProfile.name,
    bio: user.bio ?? fallbackProfile.bio,
    avatarUrl: user.avatarUrl ?? fallbackProfile.avatarUrl,
    url: user.url ?? fallbackProfile.url,
    company: user.company ?? fallbackProfile.company,
    location: user.location ?? fallbackProfile.location,
    followers: user.followers?.totalCount ?? fallbackProfile.followers,
    following: user.following?.totalCount ?? fallbackProfile.following,
    publicRepos: user.repositories?.totalCount ?? fallbackProfile.publicRepos,
    stars: user.starredRepositories?.totalCount ?? fallbackProfile.stars,
    repos: pinnedRepos.length > 0 ? pinnedRepos : fallbackProfile.repos,
  };
};

const fetchRESTProfile = async (): Promise<GitHubProfile> => {
  const headers = {
    Accept: 'application/vnd.github+json',
    'User-Agent': userAgent,
  };

  const [userResponse, reposResponse, starredResponse] = await Promise.all([
    fetch('https://api.github.com/users/zzqDeco', { headers }),
    fetch('https://api.github.com/users/zzqDeco/repos?per_page=100&sort=updated', { headers }),
    fetch('https://api.github.com/users/zzqDeco/starred?per_page=1', { headers }),
  ]);

  if (!userResponse.ok || !reposResponse.ok) {
    throw new Error('REST profile request failed.');
  }

  const user = await userResponse.json();
  const repos = await reposResponse.json();
  const stars = parseLastPage(starredResponse.headers.get('link')) ?? fallbackProfile.stars;

  const representativeRepos = repos
    .filter((repo: any) => !repo.fork && !repo.archived)
    .slice(0, pinnedRepoLimit)
    .map(toRepo);

  return {
    login: user.login ?? fallbackProfile.login,
    name: user.name ?? user.login ?? fallbackProfile.name,
    bio: user.bio ?? fallbackProfile.bio,
    avatarUrl: user.avatar_url ?? fallbackProfile.avatarUrl,
    url: user.html_url ?? fallbackProfile.url,
    company: user.company ?? fallbackProfile.company,
    location: user.location ?? fallbackProfile.location,
    followers: user.followers ?? fallbackProfile.followers,
    following: user.following ?? fallbackProfile.following,
    publicRepos: user.public_repos ?? fallbackProfile.publicRepos,
    stars,
    repos: representativeRepos.length > 0 ? representativeRepos : fallbackProfile.repos,
  };
};

export const getGitHubProfile = async (): Promise<GitHubProfile> => {
  try {
    const token = import.meta.env.GITHUB_TOKEN ?? process.env.GITHUB_TOKEN;
    if (token) {
      return await fetchGraphQLProfile(token);
    }
  } catch {
    // Fall through to REST. Profile data is decorative and must not fail the site build.
  }

  try {
    return await fetchRESTProfile();
  } catch {
    return fallbackProfile;
  }
};

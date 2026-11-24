import fs from 'fs'
import os from 'os'
import path from 'path'
import { beforeEach, expect, it, vi } from 'vitest'

const driver = vi.hoisted(() => ({ listImages:vi.fn(),listContainers: vi.fn(), getContainer: vi.fn(), stop: vi.fn(), remove: vi.fn() }))
vi.mock('dockerode', () => ({ default: class {
  listImages=driver.listImages
  listContainers = driver.listContainers
  getContainer = driver.getContainer
} }))
import { DockerManager, dockerBuildId } from '../../electron/docker'

beforeEach(() => {
  vi.clearAllMocks()
  driver.getContainer.mockReturnValue({ stop: driver.stop, remove: driver.remove })
})

it('does not delete a container with a similar name', async () => {
  driver.listContainers.mockResolvedValue([
    { Id: 'other', Names: ['/old-mah-studio-backend'], State: 'running', Labels: {} },
    { Id: 'owned', Names: ['/mah-studio-backend'], State: 'running', Labels: { 'com.modelsathome.studio': '1' } },
  ])
  await new DockerManager('/tmp/runtime').ensureContainerRemoved()
  expect(driver.getContainer).toHaveBeenCalledExactlyOnceWith('owned')
  expect(driver.remove).toHaveBeenCalledOnce()
})

it('refuses to replace an unmanaged container even with the exact name', async () => {
  driver.listContainers.mockResolvedValue([{ Id: 'unmanaged', Names: ['/mah-studio-backend'], State: 'running', Labels: {} }])
  await expect(new DockerManager('/tmp/runtime').ensureContainerRemoved()).rejects.toThrow('unmanaged container')
  expect(driver.stop).not.toHaveBeenCalled()
  expect(driver.remove).not.toHaveBeenCalled()
})


it('requires an image matching current backend/runtime and ignores mutable data',async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'mah-docker-build-'))
  try{
    for(const dir of ['backend','examples','models-at-home/homellm']) fs.mkdirSync(path.join(root,dir),{recursive:true})
    for(const file of ['requirements-world-foundation.txt','requirements-world-local.txt','models-at-home/LICENSE','models-at-home/Dockerfile','models-at-home/requirements.txt','models-at-home/pyproject.toml','backend/api.py']) fs.writeFileSync(path.join(root,file),'source')
    const id=dockerBuildId(root);const manager=new DockerManager(path.join(root,'models-at-home'),root)
    driver.listImages.mockResolvedValue([{Labels:{}}]);expect(await manager.imageExists()).toBe(false)
    driver.listImages.mockResolvedValue([{Labels:{'com.modelsathome.studio.build':id}}]);expect(await manager.imageExists()).toBe(true)
    fs.mkdirSync(path.join(root,'models-at-home/models'));fs.writeFileSync(path.join(root,'models-at-home/models/user-weight.bin'),'user')
    fs.mkdirSync(path.join(root,'backend/__pycache__'));fs.writeFileSync(path.join(root,'backend/__pycache__/api.pyc'),'cache')
    expect(dockerBuildId(root)).toBe(id)
    fs.writeFileSync(path.join(root,'models-at-home/requirements.txt'),'updated deps');expect(await manager.imageExists()).toBe(false)
    fs.writeFileSync(path.join(root,'models-at-home/requirements.txt'),'source')
    fs.writeFileSync(path.join(root,'backend/api.py'),'updated');expect(await manager.imageExists()).toBe(false)
  }finally{fs.rmSync(root,{recursive:true,force:true})}
})
